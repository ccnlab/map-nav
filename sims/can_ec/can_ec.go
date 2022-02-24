// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
v1rf illustrates how self-organizing learning in response to natural images
produces the oriented edge detector receptive field properties of neurons
in primary visual cortex (EC). This provides insight into why the visual
system encodes information in the way it does, while also providing an
important test of the biological relevance of our computational models.
*/
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/emer/etable/agg"

	"github.com/emer/empi/mpi"

	"github.com/goki/gi/gist"

	"github.com/emer/emergent/actrf"
	"github.com/emer/emergent/edge"
	"github.com/emer/emergent/efuns"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

func main() {
	TheSim.New() // note: not running Config here -- done in CmdArgs for mpi / nogui
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		TheSim.Config()      // for GUI case, config then run..
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	fwin := TheSim.ConfigWorldGui()
	fwin.GoStartEventLoop()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no extra learning factors, hebbian learning",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "false", // works well without
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.WtBal.On":    "true", // this is typically critical
					// "Prjn.Learn.XCal.MLrn":    "0", // pure hebb -- NO!  doing edl
					// "Prjn.Learn.XCal.SetLLrn": "true",
					// "Prjn.Learn.XCal.LLrn":    "1",
					"Prjn.Learn.WtSig.Gain": "6", // 6 impedes learning..  was 1
					"Prjn.Learn.Learn":      "true",
					//"Prjn.WtInit.Mean":        "0.5",
					//"Prjn.WtInit.Var":         "0.0", // even .01 causes some issues..
				}},
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					//"Layer.Learn.AvgL.Gain":   "1", // this is critical! much lower
					//"Layer.Learn.AvgL.Min":    "0.01",
					//"Layer.Learn.AvgL.Init":   "0.2",
					"Layer.Inhib.Layer.Gi":    "2.0", // more active..
					"Layer.Inhib.Layer.FBTau": "3",
					"Layer.Inhib.ActAvg.Init": "0.1",
					"Layer.Act.Gbar.L":        "0.1",
					"Layer.Act.Dt.GTau":       "3", // slower = more noise integration -- otherwise fails sometimes
					//"Layer.Act.Noise.Dist":    "Gaussian",
					//"Layer.Act.Noise.Var":     "0.004", // 0.002 fails to converge sometimes, .005 a bit noisy
					//"Layer.Act.Noise.Type":    "GeNoise",
					//"Layer.Act.Noise.Fixed":   "false",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
				}},
			{Sel: ".ExciteLateral", Desc: "lateral excitatory connection",
				Params: params.Params{
					//"Prjn.Off":         "true",
					"Prjn.Learn.Learn": "true",
					"Prjn.WtInit.Mean": "0.5",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
					"Prjn.WtScale.Rel": "0.2", // this controls the speed -- higher = faster
				}},
			{Sel: ".InhibLateral", Desc: "lateral inhibitory connection",
				Params: params.Params{
					//"Prjn.Off":         "true",
					"Prjn.Learn.Learn": "true",
					"Prjn.WtInit.Mean": "0.5",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
					"Prjn.WtScale.Abs": "0.5", // higher gives better grid
				}},
			{Sel: ".OrientationForward", Desc: "orientation to ec forward connection",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.1",
				}},
			{Sel: "#EC", Desc: "all EC layers: only pools, no layer-level",
				Params: params.Params{
					//"Layer.Act.Init.Decay": "0",
					"Layer.Act.Noise.Dist":    "Gaussian",
					"Layer.Act.Noise.Var":     "0.0", // 0.004, 0.002 fails to converge sometimes, .005 a bit noisy
					"Layer.Act.Noise.Type":    "GeNoise",
					"Layer.Act.Noise.Fixed":   "false",
					"Layer.Inhib.ActAvg.Init": "0.08",
					"Layer.Inhib.Layer.Gi":    "1.8",
				}},
			{Sel: ".Position", Desc: "position layers",
				Params: params.Params{
					// "Layer.Act.Init.Decay":    "0",
					"Layer.Inhib.Layer.Gi":    "2.0",  // for EC = 20
					"Layer.Inhib.ActAvg.Init": "0.05", // it is essential to set this for all layers
				}},
			{Sel: ".Orientation", Desc: "orientation layers",
				Params: params.Params{
					// "Layer.Act.Init.Decay":    "0",
					"Layer.Inhib.Layer.Gi":    "2.0",
					"Layer.Inhib.ActAvg.Init": "0.15", // it is essential to set this for all layers
				}},
			{Sel: "#ECToOut_Position", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn": "true",
					"Prjn.WtInit.Var":  "0.25",
					"Prjn.WtInit.Sym":  "false", // actually works better non-sym

					// hip_bench setting
					//"Prjn.Learn.Learn": "false", // learning here definitely does NOT work!
					//"Prjn.WtInit.Mean": "0.9",
					//"Prjn.WtInit.Var":  "0.01",
					//"Prjn.WtScale.Rel": "4",
				}},
			{Sel: "#ECToOrientation", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn": "true",
					"Prjn.WtInit.Var":  "0.25",
					"Prjn.WtInit.Sym":  "false",
				}},
			//{Sel: "#Out_PositionToEC", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
			//	Params: params.Params{
			//		"Prjn.Learn.Learn": "true",
			//		"Prjn.WtInit.Var":  "0.25",
			//		"Prjn.WtScale.Rel": "1",
			//	}},
			{Sel: "#OrientationToEC", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn": "true",
					"Prjn.WtInit.Var":  "0.25",
					"Prjn.WtScale.Rel": ".1", // orientation is easier so give it a weaker top-down err
				}},
			{Sel: "#VestibularToEC", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn": "true",
					"Prjn.WtInit.Var":  "0.25",
				}},
			{Sel: "#Prev_PositionToEC", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn": "true",
					//"Prjn.WtInit.Var":  "0",
				}},
			{Sel: "#Prev_OriToEC", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn": "true",
					//"Prjn.WtInit.Var":  "0",
				}},
			//{Sel: "#Out_PositionToOut_Position", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
			//	Params: params.Params{
			//		//"Prjn.Learn.Learn": "false",
			//		//"Prjn.WtInit.Var":  "0",
			//		//"Prjn.WtInit.Mean": "0.8",
			//		//"Prjn.WtScale.Rel": "0.1", // zycyc experiment
			//
			//		// hip_bench setting
			//		"Prjn.Learn.Momentum.On": "false",
			//		"Prjn.Learn.Norm.On":     "false",
			//		"Prjn.Learn.WtBal.On":    "true",
			//	}},
			//{Sel: "#OrientationToOrientation", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
			//	Params: params.Params{
			//		"Prjn.Learn.Learn": "false",
			//		"Prjn.WtInit.Var":  "0",
			//		"Prjn.WtInit.Mean": "0.8",
			//		"Prjn.WtScale.Rel": "0.1", // zycyc experiment
			//	}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Entorhinal EcParams     `desc:"EC sizing parameters"`
	Pat        PatParams    `desc:"parameters for the input patterns"`
	PoolVocab  patgen.Vocab `view:"no-inline" desc:"pool patterns vocabulary"`
	//ExcitLateralScale float32           `def:"0.2" desc:"excitatory lateral (recurrent) WtScale.Rel value"`
	//InhibLateralScale float32           `def:"0.2" desc:"inhibitory lateral (recurrent) WtScale.Abs value"`
	//ExcitLateralLearn bool              `def:"true" desc:"do excitatory lateral (recurrent) connections learn?"`
	Net              *leabra.Network  `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	CorticalInput    *etable.Table    `view:"no-inline" desc:"input patterns generated"`
	OrientationInput *etable.Table    `view:"no-inline" desc:"input patterns generated"`
	Probes           *etable.Table    `view:"no-inline" desc:"probe inputs"`
	ARFs             actrf.RFs        `view:"no-inline" desc:"activation-based receptive fields"`
	TrnTrlLog        *etable.Table    `view:"no-inline" desc:"training trial-level log data"`
	TrnEpcLog        *etable.Table    `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog        *etable.Table    `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog        *etable.Table    `view:"no-inline" desc:"testing trial-level log data"`
	RunLog           *etable.Table    `view:"no-inline" desc:"summary log of each run"`
	RunStats         *etable.Table    `view:"no-inline" desc:"aggregate stats on all runs"`
	Params           params.Sets      `view:"no-inline" desc:"full collection of param sets"`
	ParamSet         string           `view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	Tag              string           `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	EConWts          *etensor.Float32 `view:"-" desc:"weights from input to EC layer"`
	ECoffWts         *etensor.Float32 `view:"-" desc:"weights from input to EC layer"`
	ECWts            *etensor.Float32 `view:"no-inline" desc:"net on - off weights from input to EC layer"`
	MaxRuns          int              `desc:"maximum number of model runs to perform"`
	MaxEpcs          int              `desc:"maximum number of epochs to run per model run"`
	TestEpcs         int              `desc:"number of epochs of testing to run, cumulative after MaxEpcs of training"`
	//MaxTrls           int               `desc:"maximum number of training trials per epoch"`
	//TrainEnv   env.FixedTable    `desc:"Training environment -- visual images"`
	Time      leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn    bool              `desc:"whether to update the network view while running"`
	TrainUpdt leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt  leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	ARFLayers []string          `desc:"names of layers to compute position activation fields on"`
	TrainEnv  XYHDEnv           `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`

	// statistics: note use float64 as that is best for etable.Table
	RFMaps        map[string]*etensor.Float32 `view:"no-inline" desc:"maps for plotting activation-based receptive fields"`
	InputLays     []string                    `view:"-" desc:"input layers"`
	TargetLays    []string                    `view:"-" desc:"target layers"`
	ActAction     string                      `inactive:"+" desc:"action generated & taken"`
	TrlCosDiff    float64                     `inactive:"+" desc:"current trial's overall cosine difference"`
	TrlCosDiffTGT []float64                   `inactive:"+" desc:"current trial's cosine difference for target layers"`
	EpcCosDiff    float64                     `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	NumTrlStats   int                         `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`

	// internal state - view:"-"
	Win                *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView            *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar            *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	WorldWin           *gi.Window                  `view:"-" desc:"XYHDEnv GUI window"`
	WorldTabs          *gi.TabView                 `view:"-" desc:"XYHDEnv TabView"`
	MatColors          []string                    `desc:"color strings in material order"`
	Trace              *etensor.Int                `view:"no-inline" desc:"trace of movement for visualization"`
	TraceView          *etview.TensorGrid          `desc:"view of the activity trace"`
	dTrace             *etensor.Int                `view:"no-inline" desc:"trace of movement for visualization"`
	dTraceView         *etview.TensorGrid          `desc:"view of the activity trace"`
	WorldView          *etview.TensorGrid          `desc:"view of the world"`
	CurImgGrid         *etview.TensorGrid          `view:"-" desc:"the current image grid view"`
	WtsGrid            *etview.TensorGrid          `view:"-" desc:"the weights grid view"`
	TrnTrlPlot         *eplot.Plot2D               `view:"-" desc:"the training trial plot"`
	TrnEpcPlot         *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot         *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot         *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	RunPlot            *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile         *os.File                    `view:"-" desc:"log file"`
	TstEpcFile         *os.File                    `view:"-" desc:"log file"`
	RunFile            *os.File                    `view:"-" desc:"log file"`
	ValsTsrs           map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	EClateralflag      bool                        `view:"-" desc:"flag for EClateral"`
	IsRunning          bool                        `view:"-" desc:"true if sim is running"`
	StopNow            bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun        bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	UseMPI             bool                        `view:"-" desc:"if true, use MPI to distribute computation across nodes"`
	SaveWts            bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	SaveARFs           bool                        `view:"-" desc:"for command-line run only, auto-save receptive field data"`
	NoGui              bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	RndSeed            int64                       `view:"-" desc:"the current random seed"`
	init_pos_assigned1 bool                        `view:"-" desc:"true if initial position pattern has been created and applied"`
	Comm               *mpi.Comm                   `view:"-" desc:"mpi communicator"`
	AllDWts            []float32                   `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts            []float32                   `view:"-" desc:"buffer of MPI summed dwt weight changes"`
}

// EcParams have the entorhinal cortex size and connectivity parameters
type EcParams struct {
	ECSize            evec.Vec2i `desc:"size of EC"`
	LstmSize          evec.Vec2i `desc:"size of EC"`
	InputSize         evec.Vec2i `desc:"size of Input"`
	PositionSize      evec.Vec2i `desc:"size of Position"`
	OrientationSize   evec.Vec2i `desc:"size of Orientation (head direction, 0-360)"`
	VestibularSize    evec.Vec2i `desc:"size of Vestibular (left, forward, right)"`
	InputPctAct       float32    `desc:"percent active in input patterns"`
	OrientationPctAct float32    `desc:"percent active in input patterns"`
	excitRadius2D     int        `desc:"excitRadius2D"` // note: note visible b/c lower case..
	inhibRadius2D     int        `desc:"inhibRadius2D"`
	excitRadius4D     int        `desc:"excitRadius4D"`
	inhibRadius4D     int        `desc:"inhibRadius4D"`
	excitSigma2D      float32    `desc:"excitSigma2D"`
	inhibSigma2D      float32    `desc:"inhibSigma2D"`
	excitSigma4D      float32    `desc:"excitSigma4D"`
	inhibSigma4D      float32    `desc:"inhibSigma4D"`
}

// PatParams have the pattern parameters
type PatParams struct {
	ListSize   int     `desc:"number of A-B, A-C patterns each"`
	MinDiffPct float32 `desc:"minimum difference between item random patterns, as a proportion (0-1) of total active"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	//ss.ExcitLateralScale = 0.2 // zycyc: larger inhib circle
	//ss.InhibLateralScale = 0.2
	//ss.ExcitLateralLearn = false // zycyc: no learning
	ss.Net = &leabra.Network{}
	ss.PoolVocab = patgen.Vocab{}
	ss.CorticalInput = &etable.Table{}
	ss.OrientationInput = &etable.Table{}
	ss.Probes = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.EConWts = &etensor.Float32{}
	ss.ECoffWts = &etensor.Float32{}
	ss.ECWts = &etensor.Float32{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.Cycle
	ss.TestUpdt = leabra.Cycle
	ss.ARFLayers = []string{"EC", "Orientation", "Out_Position"}
	ss.init_pos_assigned1 = false
	ss.EClateralflag = true

	ss.Entorhinal.Defaults()
	ss.Pat.Defaults()
}

func (ec *EcParams) Defaults() {
	ec.ECSize.Set(20, 20) // 30 needs lower Pos Gi, but just slightly better compared to 20
	//ec.LstmSize.Set(12, 12)
	//ec.InputSize.Set(12, 12) // zycyc: ?? automatic!
	ec.PositionSize.Set(12, 12) // Gi needs to change as well??
	ec.OrientationSize.Set(16, 1)
	ec.VestibularSize.Set(12, 1)
	ec.InputPctAct = 0.25
	ec.OrientationPctAct = 0.25

	//ec.excitRadius2D = 5
	//ec.excitSigma2D = 3
	//ec.inhibRadius2D = 10
	//ec.inhibSigma2D = 10

	ec.excitRadius4D = 3 // was 3, 1
	ec.excitSigma4D = 2
	ec.inhibRadius4D = 8 // was 8 (Pos Gi 3.2 works; 3.6 also works?), 5 (Pos Gi 3.6 works)
	ec.inhibSigma4D = 2  // not really sure what this should be, seems like as long as it's not too small it's fine, 2 looks best
}

func (pp *PatParams) Defaults() {
	pp.ListSize = 10 // 10 is too small to see issues..
	pp.MinDiffPct = 0.5
	//pp.CtxtFlipPct = .25
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	//ss.OpenPats()
	//ss.ConfigPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 1
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 300  // zycyc, test
		ss.TestEpcs = 500 // zycyc, test
	}
	//if ss.MaxTrls == 0 { // allow user override
	//	ss.MaxTrls = 100
	//}

	ss.TrainEnv.Config(300) // 1000) // n trials per epoch
	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TrainEnv.Init(0)
	ss.TrainEnv.Validate()

	ss.ConfigRFMaps()
}

func (ss *Sim) ConfigRFMaps() {
	ss.RFMaps = make(map[string]*etensor.Float32)
	mt := &etensor.Float32{}
	mt.CopyShapeFrom(ss.TrainEnv.World)
	ss.RFMaps["Pos"] = mt

	mt = &etensor.Float32{}
	mt.SetShape([]int{len(ss.TrainEnv.Acts)}, nil, nil)
	ss.RFMaps["Act"] = mt

	mt = &etensor.Float32{}
	mt.SetShape([]int{ss.TrainEnv.NRotAngles}, nil, nil)
	ss.RFMaps["Ang"] = mt

	mt = &etensor.Float32{}
	mt.SetShape([]int{3}, nil, nil)
	ss.RFMaps["Rot"] = mt
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	ecParam := &ss.Entorhinal
	net.InitName(net, "can_ec")
	prevPosition := net.AddLayer2D("Prev_Position", ecParam.PositionSize.Y, ecParam.PositionSize.X, emer.Input)
	prevPosition.SetClass("Position")
	prevOri := net.AddLayer2D("Prev_Ori", ecParam.OrientationSize.Y, ecParam.OrientationSize.X, emer.Input)
	prevOri.SetClass("Orientation")

	vestibular := net.AddLayer2D("Vestibular", ecParam.VestibularSize.Y, ecParam.VestibularSize.X, emer.Input)
	vestibular.SetClass("Orientation")
	//lstm := net.AddLayer2D("LSTM", ecParam.LstmSize.Y, ecParam.LstmSize.X, emer.Hidden)
	ec := net.AddLayer4D("EC", ecParam.ECSize.Y, ecParam.ECSize.X, 2, 2, emer.Hidden)
	// ec := net.AddLayer2D("EC", 16, 16, emer.Hidden)

	outPosition := net.AddLayer2D("Out_Position", ecParam.PositionSize.Y, ecParam.PositionSize.X, emer.Target)
	outPosition.SetClass("Position")

	orientation := net.AddLayer2D("Orientation", ecParam.OrientationSize.Y, ecParam.OrientationSize.X, emer.Target)
	orientation.SetClass("Orientation")

	//////////////////////////////////////////// EC first for indexing convinience
	//ec := net.AddLayer2D("EC", ecParam.ECSize.Y, ecParam.ECSize.X, emer.Hidden) // 2D EC

	// 2D EC
	//excit := prjn.NewCircle()
	//excit.TopoWts = true
	//excit.Radius = ecParam.excitRadius2D
	//excit.Sigma = ecParam.excitSigma2D

	//inhib := prjn.NewCircle()
	//inhib.TopoWts = true
	//inhib.Radius = ecParam.inhibRadius2D
	//inhib.Sigma = ecParam.inhibSigma2D

	//rect := prjn.NewRect()
	//rect.Size.Set(1, 1)
	//orie := net.ConnectLayers(orientation, ec, rect, emer.Forward)
	//orie.SetClass("OrientationForward")

	// 4D EC
	//excit := prjn.NewCircle()
	//excit.TopoWts = true
	//excit.Radius = ecParam.excitRadius4D
	//excit.Sigma = ecParam.excitSigma4D

	excit := prjn.NewPoolTile()
	excit.Size.Set(2*ecParam.excitRadius4D+1, 2*ecParam.excitRadius4D+1)
	excit.Skip.Set(1, 1)
	excit.Start.Set(-ecParam.excitRadius4D, -ecParam.excitRadius4D)
	excit.TopoRange.Min = 0.8
	excit.GaussInPool.On = false

	inhib := prjn.NewCircle()
	inhib.TopoWts = true
	inhib.Radius = ecParam.inhibRadius4D
	inhib.Sigma = ecParam.inhibSigma4D

	// inhib := prjn.NewPoolTile()
	// inhib.Size.Set(2*ecParam.inhibRadius4D+1, 2*ecParam.inhibRadius4D+1)
	// inhib.Skip.Set(1, 1)
	// inhib.Start.Set(-ecParam.inhibRadius4D, -ecParam.inhibRadius4D)
	// inhib.GaussInPool.On = false
	// inhib.TopoRange.Min = 0.8

	// original orientation to ec prjn
	//oriePrjn := prjn.NewPoolSameUnit()
	//orie := net.ConnectLayers(orientation, ec, oriePrjn, emer.Forward)
	//orie.SetClass("OrientationForward")

	rec := net.ConnectLayers(ec, ec, excit, emer.Lateral)
	rec.SetClass("ExciteLateral")

	//inh := net.ConnectLayers(ec, ec, full, emer.Inhib)
	inh := net.ConnectLayers(ec, ec, inhib, emer.Inhib)
	inh.SetClass("InhibLateral")

	//////////////////////////////////////////// other connections
	full := prjn.NewFull()
	//oriePrjn := prjn.NewPoolSameUnit()
	//random := prjn.NewUnifRnd()

	//net.ConnectLayers(prevPosition, lstm, full, emer.Forward)
	//net.ConnectLayers(prevOri, lstm, full, emer.Forward)
	//net.ConnectLayers(vestibular, lstm, full, emer.Forward)
	//net.ConnectLayers(lstm, ec, full, emer.Forward)

	net.ConnectLayers(prevPosition, ec, full, emer.Forward)
	net.ConnectLayers(prevOri, ec, full, emer.Forward)
	net.ConnectLayers(vestibular, ec, full, emer.Forward)
	net.BidirConnectLayers(ec, outPosition, full)
	net.BidirConnectLayers(ec, orientation, full)

	//one2one := prjn.NewOneToOne()
	//net.LateralConnectLayer(outPosition, full)
	//net.ConnectLayers(prevPosition, outPosition, full, emer.Forward)

	//net.LateralConnectLayer(orientation, one2one)

	prevOri.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Prev_Position", YAlign: relpos.Front, Space: 2})
	vestibular.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Prev_Ori", YAlign: relpos.Front, Space: 2})
	//lstm.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Prev_Position", XAlign: relpos.Left, YAlign: relpos.Front, Space: 0})
	ec.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Prev_Position", XAlign: relpos.Left, YAlign: relpos.Front, Space: 0})
	outPosition.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "EC", XAlign: relpos.Left, YAlign: relpos.Front, Space: 0})
	orientation.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Out_Position", YAlign: relpos.Front, Space: 2})

	//////////////////////////////////////
	// collect

	if len(ss.InputLays) == 0 && len(ss.TargetLays) == 0 {
		ss.InputLays = make([]string, 0, 10)
		ss.TargetLays = make([]string, 0, 10)
		for _, ly := range net.Layers {
			if ly.IsOff() {
				continue
			}
			switch ly.Type() {
			case emer.Input:
				ss.InputLays = append(ss.InputLays, ly.Name())
			case emer.Target:
				ss.TargetLays = append(ss.TargetLays, ly.Name())
			}
		}
	}

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

func (ss *Sim) ReConfigNet() {
	//ss.ConfigPats()
	ss.Net = &leabra.Network{} // start over with new network
	ss.ConfigNet(ss.Net)
	if ss.NetView != nil {
		ss.NetView.SetNet(ss.Net)
		ss.NetView.Update() // issue #41 closed
	}
}

func (ss *Sim) InitWts(net *leabra.Network) {
	net.InitTopoScales() // needed for gaussian topo Circle wts
	net.InitWts()
	if ss.EClateralflag {
		ss.InitLateralWts(net)
	}
}

func (ss *Sim) InitLateralWts(net *leabra.Network) {
	ecParam := &ss.Entorhinal
	ec := net.LayerByName("EC").(leabra.LeabraLayer).AsLeabra()
	lat := ec.RecvPrjn(0) // ?? zycyc: fix this
	nPy := ec.Shape().Dim(0)
	nPx := ec.Shape().Dim(1)
	//radius := ecParam.excitRadius2D // 2D EC
	radius := ecParam.excitRadius4D // 4D EC

	offsets := []float32{1, 1, -1, 1, 1, -1, -1, -1} // four corners
	//offsets := []float32{0, -1, 1, 0, -1, 0, 0, 1} // up, down, left, right

	// 4D EC
	for py := 0; py < nPy; py++ {
		for px := 0; px < nPx; px++ {
			sri := (py*nPx + px) * 4
			for i := 0; i < 4; i++ { // 4 receiving units
				ri := sri + i
				rctr := mat32.Vec2{offsets[i*2], offsets[i*2+1]} // center of gaussian for recv unit
				for sy := -radius; sy <= radius; sy++ {          // circle
					for sx := -radius; sx <= radius; sx++ {
						for j := 0; j < 4; j++ { // 4 sending units
							spy, _ := edge.Edge(py+sy, nPy, true)
							spx, _ := edge.Edge(px+sx, nPx, true)
							si := (spy*nPx+spx)*4 + j
							v := mat32.NewVec2(float32(sx), float32(sy))
							wt := efuns.GaussVecDistNoNorm(v, rctr, ecParam.excitSigma4D)
							lat.SetSynVal("Wt", si, ri, wt)
						}
					}
				}
			}
		}
	}

	// 2D EC
	//for py := 0; py < nPy; py++ {
	//	for px := 0; px < nPx; px++ {
	//		ri := py * nPx + px
	//		for sy := -radius; sy <= radius; sy++ { // circle
	//			for sx := -radius; sx <= radius; sx++ {
	//				spy, _ := edge.Edge(py + sy, nPy, true)
	//				spx, _ := edge.Edge(px + sx, nPx, true)
	//				si := spy * nPx + spx
	//				v := mat32.NewVec2(float32(sx) - float32(ss.Xisign(spx, spy)), float32(sy) - float32(ss.Yisign(spx, spy)))
	//				d := v.Length()
	//				wt := efuns.Gauss1DNoNorm(d, ecParam.excitSigma2D)
	//				lat.SetSynVal("Wt", si, ri, wt)
	//			}
	//		}
	//	}
	//}
}

// 2D EC
//func (ss *Sim) Xisign(spx, spy int) int {
//	if spx % 2 == 0 && spy % 2 == 0 {
//		return 1
//	} else if spx % 2 == 0 && spy % 2 != 0{
//		return 1
//	} else if spx % 2 != 0 && spy % 2 == 0 {
//		return -1
//	} else {
//		return -1
//	}
//}
//
//func (ss *Sim) Yisign(spx, spy int) int {
//	if spx % 2 == 0 && spy % 2 == 0 {
//		return 1
//	} else if spx % 2 == 0 && spy % 2 != 0{
//		return -1
//	} else if spx % 2 != 0 && spy % 2 == 0 {
//		return 1
//	} else {
//		return -1
//	}
//}
//
//// 4D EC
//func (ss *Sim) Xisign(i int) int {
//	if i == 0 || i == 2 {
//		return 1
//	} else {
//		return -1
//	}
//}
//
//func (ss *Sim) Yisign(i int) int {
//	if i < 2 {
//		return 1
//	} else {
//		return -1
//	}
//}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.StopNow = false
	ss.init_pos_assigned1 = false
	ss.SetParams("", false) // all sheets
	ss.ReConfigNet()
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	ss.NewRun()
	ss.UpdateView(true)

}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		//return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tEvent:\t%d\tCycle:\t%d\tAct:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Event.Cur, ss.Time.Cycle, ss.ActAction)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Event.Cur, ss.Time.Cycle, ss.ActAction)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
	ss.UpdateWorldGui()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}

	// note: this is bad!  setting the input projections to 0!
	// ec := ss.Net.LayerByName("EC").(leabra.LeabraLayer).AsLeabra()
	// if ss.EClateralflag {
	// 	ec.Act.Clamp.Hard = false
	// } else {
	// 	late := ec.RecvPrjn(0).(leabra.LeabraPrjn).AsLeabra() // ?? zycyc: fix this
	// 	lati := ec.RecvPrjn(1).(leabra.LeabraPrjn).AsLeabra() // ?? zycyc: fix this
	// 	late.WtScale.Rel = 0
	// 	lati.WtScale.Rel = 0
	// }

	//pos := ss.Net.LayerByName("Out_Position").(leabra.LeabraLayer).AsLeabra()
	//ori := ss.Net.LayerByName("Orientation").(leabra.LeabraLayer).AsLeabra()
	//
	//PosFmEC := pos.RcvPrjns.SendName("EC").(leabra.LeabraPrjn).AsLeabra()
	//OriFmEC := ori.RcvPrjns.SendName("EC").(leabra.LeabraPrjn).AsLeabra()
	//PosFmEC.WtScale.Rel = 0
	//OriFmEC.WtScale.Rel = 0
	//
	//PosFmPos := pos.RcvPrjns.SendName("Out_Position").(leabra.LeabraPrjn).AsLeabra()
	//OriFmOri := ori.RcvPrjns.SendName("Orientation").(leabra.LeabraPrjn).AsLeabra()
	//PosFmPos.WtScale.Rel = 1
	//OriFmOri.WtScale.Rel = 1

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()

	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		//switch qtr + 1 {
		//case 1: // Third Quarters: CA1 is driven by CA3 recall
		//	PosFmEC.WtScale.Rel = 1
		//	//OriFmEC.WtScale.Rel = 1
		//	//PosFmPos.WtScale.Rel = 0
		//	//OriFmOri.WtScale.Rel = 0
		//	ss.Net.GScaleFmAvgAct() // update computed scaling factors
		//	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		//}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		//ss.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
}

//// QuarterInc increments at the quarter level, updating Quarter and PlusPhase
//func (ss *Sim) QuarterInc() {
//	tm := &ss.Time
//	tm.Quarter++
//	if tm.Quarter == 5 {
//		tm.PlusPhase = true
//	} else {
//		tm.PlusPhase = false
//	}
//}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) TakeAction(net *leabra.Network, ev *XYHDEnv) {
	gact := ev.ActGen()
	ss.ActAction = ev.Acts[gact]
	ev.Action(ss.ActAction, nil)

	// fmt.Printf("action: %s\n", ev.Acts[act])
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	//ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	states := []string{"Vestibular", "Position", "Angle", "PrevPosition", "PrevAngle"}
	lays := []string{"Vestibular", "Out_Position", "Orientation", "Prev_Position", "Prev_Ori"} // zycyc: input: 16*16; orientation: 1*16 ring????
	//if !ss.init_pos_assigned1 {
	//	prevPosition := ss.Net.LayerByName("Prev_Position").(leabra.LeabraLayer).AsLeabra()
	//	prevPosition.ApplyExt(en.State("Position"))
	//	ss.init_pos_assigned1 = true
	//}

	for i, lnm := range lays {
		lyi := ss.Net.LayerByName(lnm)
		if lyi == nil {
			continue
		}
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(states[i])

		//pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TakeAction(ss.Net, &ss.TrainEnv) // zycyc: ??
	ss.TrainEnv.Step()                  // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}

		if epc >= ss.MaxEpcs {
			if ss.SaveWts { // doing this earlier
				ss.SaveWeights()
			}
			// done with training..
			if ss.SaveARFs {
				ss.TestAll()
			}
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
	if ss.CurImgGrid != nil {
		ss.CurImgGrid.UpdateSig()
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveARFs {
		ss.SaveAllARFs()
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	//ss.TrainEnv.Table = etable.NewIdxView(ss.OrientationInput)
	ss.TrainEnv.Init(run)
	ss.Time.Reset()
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.TrnTrlLog.SetNumRows(0)
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	ss.NumTrlStats = 0
	ss.TrlCosDiff = 0
	ss.SumCosDiff = 0
	ss.EpcCosDiff = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	nt := len(ss.TargetLays)
	if len(ss.TrlCosDiffTGT) != nt {
		ss.TrlCosDiffTGT = make([]float64, nt)
	}
	acd := 0.0
	for i, ln := range ss.TargetLays {
		ly := ss.Net.LayerByName(ln).(leabra.LeabraLayer).AsLeabra()
		cd := float64(ly.CosDiff.Cos)
		acd += cd
		ss.TrlCosDiffTGT[i] = cd
	}
	ss.TrlCosDiff = acd / float64(len(ss.TargetLays))
	if accum {
		ss.SumCosDiff += ss.TrlCosDiff
	}

	if accum {
		// zycyc: ?? do all the trn epc log thing
		ss.NumTrlStats++
	} else {
		ss.UpdtARFs()
	}

	return
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights() {
	fnm := ss.WeightsFileName()
	fmt.Printf("Saving Weights to: %v\n", fnm)
	ss.Net.SaveWtsJSON(gi.FileName(fnm))
}

// OpenRec2Wts opens trained weights w/ rec=0.2
//func (ss *Sim) OpenRec2Wts() {
//	ab, err := Asset("v1rf_rec2.wts") // embedded in executable
//	if err != nil {
//		log.Println(err)
//	}
//	ss.Net.ReadWtsJSON(bytes.NewBuffer(ab))
//	// ss.Net.OpenWtsJSON("v1rf_rec2.wts.gz")
//}

//// OpenRec05Wts opens trained weights w/ rec=0.05
//func (ss *Sim) OpenRec05Wts() {
//	ab, err := Asset("v1rf_rec05.wts") // embedded in executable
//	if err != nil {
//		log.Println(err)
//	}
//	ss.Net.ReadWtsJSON(bytes.NewBuffer(ab))
//	// ss.Net.OpenWtsJSON("v1rf_rec05.wts.gz")
//}

//func (ss *Sim) ECRFs() {
//	onVals := ss.EConWts.Values
//	//offVals := ss.ECoffWts.Values
//	netVals := ss.ECWts.Values
//	on := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra() // zycyc: ??
//	//off := ss.Net.LayerByName("LGNoff").(leabra.LeabraLayer).AsLeabra()
//	isz := on.Shape().Len()
//	ec := ss.Net.LayerByName("EC").(leabra.LeabraLayer).AsLeabra()
//	ysz := ec.Shape().Dim(0)
//	xsz := ec.Shape().Dim(1)
//	for y := 0; y < ysz; y++ {
//		for x := 0; x < xsz; x++ {
//			ui := (y*xsz + x)
//			ust := ui * isz
//			onvls := onVals[ust : ust+isz]
//			//offvls := offVals[ust : ust+isz]
//			netvls := netVals[ust : ust+isz]
//			on.SendPrjnVals(&onvls, "Wt", ec, ui, "")
//			//off.SendPrjnVals(&offvls, "Wt", ec, ui, "")
//			for ui := 0; ui < isz; ui++ {
//				//netvls[ui] = 1.5 * (onvls[ui] - offvls[ui])
//				netvls[ui] = 1.5 * (onvls[ui]) // zycyc: not sure what this is ??
//			}
//		}
//	}
//	if ss.WtsGrid != nil {
//		ss.WtsGrid.UpdateSig()
//	}
//}

func (ss *Sim) ConfigWts(dt *etensor.Float32) {
	dt.SetShape([]int{14, 14, 12, 12}, nil, nil)
	dt.SetMetaData("grid-fill", "1")
}

// SetAFMetaData
func (ss *Sim) SetAFMetaData(af etensor.Tensor) {
	af.SetMetaData("min", "0")
	af.SetMetaData("colormap", "Viridis") // "JetMuted")
	af.SetMetaData("grid-fill", "1")
}

// UpdtARFs updates position activation rf's
func (ss *Sim) UpdtARFs() {
	for nm, mt := range ss.RFMaps {
		mt.SetZeros()
		switch nm {
		case "Pos":
			mt.Set([]int{ss.TrainEnv.PosI.Y, ss.TrainEnv.PosI.X}, 1)
		case "Act":
			mt.Set1D(ss.TrainEnv.Act, 1)
		case "Ang":
			mt.Set1D(ss.TrainEnv.Angle/90, 1)
		case "Rot":
			mt.Set1D(1+ss.TrainEnv.RotAng/90, 1)
		}
	}

	naf := len(ss.ARFLayers) * len(ss.RFMaps)
	if len(ss.ARFs.RFs) != naf {
		for _, lnm := range ss.ARFLayers {
			ly := ss.Net.LayerByName(lnm)
			if ly == nil {
				continue
			}
			vt := ss.ValsTsr(lnm)
			ly.UnitValsTensor(vt, "ActM")
			for nm, mt := range ss.RFMaps {
				af := ss.ARFs.AddRF(lnm+"_"+nm, vt, mt)
				ss.SetAFMetaData(&af.NormRF)
			}
		}
	}
	for _, lnm := range ss.ARFLayers {
		ly := ss.Net.LayerByName(lnm)
		if ly == nil {
			continue
		}
		vt := ss.ValsTsr(lnm)
		ly.UnitValsTensor(vt, "ActM")
		for nm, mt := range ss.RFMaps {
			ss.ARFs.Add(lnm+"_"+nm, vt, mt, 0.01) // thr prevent weird artifacts
		}
	}
}

// SaveAllARFs saves all ARFs to files
func (ss *Sim) SaveAllARFs() {
	ss.ARFs.Avg()
	ss.ARFs.Norm()
	for _, paf := range ss.ARFs.RFs {
		fnm := ss.LogFileName(paf.Name)
		etensor.SaveCSV(&paf.NormRF, gi.FileName(fnm), '\t')
	}
}

// OpenAllARFs open all ARFs from directory of given path
func (ss *Sim) OpenAllARFs(path gi.FileName) {
	ss.UpdtARFs()
	ss.ARFs.Avg()
	ss.ARFs.Norm()
	ap := string(path)
	if strings.HasSuffix(ap, ".tsv") {
		ap, _ = filepath.Split(ap)
	}
	vp := ss.Win.Viewport
	for _, paf := range ss.ARFs.RFs {
		fnm := filepath.Join(ap, ss.LogFileName(paf.Name))
		err := etensor.OpenCSV(&paf.NormRF, gi.FileName(fnm), '\t')
		if err != nil {
			fmt.Println(err)
		} else {
			etview.TensorGridDialog(vp, &paf.NormRF, giv.DlgOpts{Title: "Act RF " + paf.Name, Prompt: paf.Name, TmpSave: nil}, nil, nil)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TakeAction(ss.Net, &ss.TrainEnv) // zycyc: ??
	ss.TrainEnv.Step()

	// Query counters FIRST
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTstEpc(ss.TstEpcLog)
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc >= ss.TestEpcs {
			ss.StopNow = true
			return
		}
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	//ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	//for {
	//	ss.TestTrial(true) // return on chg, don't present
	//	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	//	if chg || ss.StopNow {
	//		break
	//	}
	//} // original code in v1rf
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TestTrial(false)
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		sps := strings.Fields(ss.ParamSet)
		for _, ps := range sps {
			err = ss.SetParamsSet(ps, sheet, setMsg)
		}
	}

	//nt := ss.Net
	//ec := nt.LayerByName("EC").(leabra.LeabraLayer).AsLeabra()
	//elat := ec.RcvPrjns[2].(*leabra.Prjn)
	//elat.WtScale.Rel = ss.ExcitLateralScale
	//elat.Learn.Learn = ss.ExcitLateralLearn
	//ilat := ec.RcvPrjns[3].(*leabra.Prjn)
	//ilat.WtScale.Abs = ss.InhibLateralScale

	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

//// OpenPatAsset opens pattern file from embedded assets
//func (ss *Sim) OpenPatAsset(dt *etable.Table, fnm, name, desc string) error {
//	dt.SetMetaData("name", name)
//	dt.SetMetaData("desc", desc)
//	ab, err := Asset(fnm)
//	if err != nil {
//		log.Println(err)
//		return err
//	}
//	err = dt.ReadCSV(bytes.NewBuffer(ab), etable.Tab)
//	if err != nil {
//		log.Println(err)
//	} else {
//		for i := 1; i < len(dt.Cols); i++ {
//			dt.Cols[i].SetMetaData("grid-fill", "0.9")
//		}
//	}
//	return err
//}
//
//func (ss *Sim) OpenPats() {
//	// patgen.ReshapeCppFile(ss.Probes, "ProbeInputData.dat", "probes.csv") // one-time reshape
//	ss.OpenPatAsset(ss.Probes, "probes.tsv", "Probes", "Probe inputs for testing")
//	// err := dt.OpenCSV("probes.tsv", etable.Tab)
//}

// zycyc hack function just for developing, will need to move to patgen at one point
func InitPatsSingle(dt *etable.Table, name, desc string, inputName []string, listSize int, ySize, xSize []int) {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{inputName[0], etensor.FLOAT32, []int{ySize[0], xSize[0]}, []string{"ySize", "xSize"}},
		//{inputName[1], etensor.FLOAT32, []int{ySize[1], xSize[1]}, []string{"ySize", "xSize"}},
		//{outputName, etensor.FLOAT32, []int{ySize, xSize}, []string{"ySize", "xSize"}},
	}, listSize)
}

// zycyc hack function just for developing, will need to move to patgen at one point
func InitPats(dt *etable.Table, name, desc string, inputName []string, listSize int, ySize, xSize []int) {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{inputName[0], etensor.FLOAT32, []int{ySize[0], xSize[0]}, []string{"ySize", "xSize"}},
		{inputName[1], etensor.FLOAT32, []int{ySize[1], xSize[1]}, []string{"ySize", "xSize"}},
		//{outputName, etensor.FLOAT32, []int{ySize, xSize}, []string{"ySize", "xSize"}},
	}, listSize)
}

// zycyc hack function just for developing, will need to move to patgen at one point
func MixPats(dt *etable.Table, mp patgen.Vocab, colName []string, poolSource []string) error {
	name := dt.MetaData["name"]
	listSize := dt.ColByName(colName[0]).Shapes()[0]

	for row := 0; row < listSize; row++ {
		for iCol := 0; iCol < len(colName); iCol++ {
			dt.SetCellString("Name", row, fmt.Sprint(name, row))
			trgPool := dt.CellTensor(colName[iCol], row)
			vocNm := poolSource[iCol]
			voc, ok := mp[vocNm]
			if !ok {
				err := fmt.Errorf("Vocab not found: %s", vocNm)
				log.Println(err.Error())
				return err
			}
			vocSize := voc.Shapes()[0]
			effIdx := row % vocSize // be safe and wrap-around to re-use patterns
			frmPool := voc.SubSpace([]int{effIdx})
			trgPool.CopyFrom(frmPool)
		}
	}
	return nil
}

//func (ss *Sim) ConfigPats() {
//	ec := &ss.Entorhinal
//	inputY := ec.InputSize.Y
//	inputX := ec.InputSize.X
//	orientationY := ec.OrientationSize.Y
//	orientationX := ec.OrientationSize.X
//	npats := ss.Pat.ListSize
//	pctAct := ec.InputPctAct
//	orientationPctAct := ec.OrientationPctAct
//	//minDiff := ss.Pat.MinDiffPct
//
//	patgen.AddVocabPermutedBinary(ss.PoolVocab, "randompats", npats, inputY, inputX, pctAct, 0)
//	patgen.AddVocabPermutedBinary(ss.PoolVocab, "randomorientation", npats, orientationY, orientationX, orientationPctAct, 0)
//
//	InitPatsSingle(ss.CorticalInput, "CorticalInput", "cortical input patterns to EC", []string{"Input"}, npats, []int{inputY}, []int{inputX})
//	MixPats(ss.CorticalInput, ss.PoolVocab, []string{"Input"}, []string{"randompats"})
//
//	InitPats(ss.OrientationInput, "OrientationInput", "only orientation input patterns to EC", []string{"Input", "Orientation"}, npats, []int{inputY, orientationY}, []int{inputX, orientationX})
//	MixPats(ss.OrientationInput, ss.PoolVocab, []string{"Input", "Orientation"}, []string{"randompats", "randomorientation"})
//
//}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts.gz"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	env := &ss.TrainEnv
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur
	row := dt.Rows
	if trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	// decode position and orientation
	pos := ss.Net.LayerByName("Out_Position").(leabra.LeabraLayer).AsLeabra()
	pos_tsr := &etensor.Float32{}
	pos_tsr.SetShape([]int{env.PosSize.Y, env.PosSize.X}, nil, []string{"Y", "X"})
	for i, val := range pos.Neurons {
		pos_tsr.Values[i] = val.ActM
	}
	dec_pos, _ := env.PopCode2d.Decode(pos_tsr)

	ori := ss.Net.LayerByName("Orientation").(leabra.LeabraLayer).AsLeabra()
	ori_tsr := make([]float32, len(ori.Neurons))
	for i, val := range ori.Neurons {
		ori_tsr[i] = val.ActM
	}
	dec_ori := env.AngCode.Decode(ori_tsr)

	// acc of decoding
	dX := math.Round(float64(dec_pos.X * (float32(env.Size.X) - 2)))
	dY := math.Round(float64(dec_pos.Y * (float32(env.Size.Y) - 2)))
	poserr := math.Sqrt(math.Pow(float64(env.PosI.X)-dX, 2) + math.Pow(float64(env.PosI.Y)-dY, 2))
	posbool := float64(env.PosI.X) == dX && float64(env.PosI.Y) == dY

	oribool := false
	if math.Round(float64(dec_ori*360)) < 0 {
		dec_ori = 1 + dec_ori
	}
	if math.Round(float64(dec_ori*360)) < float64(env.AngInc)/2 || math.Abs(math.Round(float64(dec_ori*360))-360) < float64(env.AngInc)/2 {
		if env.Angle == 360 || env.Angle == 0 {
			dec_ori = float32(env.Angle)
			oribool = true
		}
	} else if math.Abs(math.Round(float64(dec_ori*360))-float64(env.Angle)) < float64(env.AngInc)/2 {
		dec_ori = float32(env.Angle)
		oribool = true
	}

	// add rows
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellFloat("Event", row, float64(env.Event.Cur))
	dt.SetCellFloat("X", row, float64(env.PosI.X))
	dt.SetCellFloat("Y", row, float64(env.PosI.Y))
	dt.SetCellFloat("dX", row, dX)
	dt.SetCellFloat("dY", row, dY)
	dt.SetCellFloat("PosErr", row, poserr)
	if posbool {
		dt.SetCellFloat("PosACC", row, float64(1))
	} else {
		dt.SetCellFloat("PosACC", row, float64(0))
	}
	dt.SetCellFloat("Ori", row, float64(env.Angle))
	if oribool {
		dt.SetCellFloat("dOri", row, float64(dec_ori))
		dt.SetCellFloat("OriErr", row, math.Abs(float64(dec_ori)-float64(env.Angle)))
		dt.SetCellFloat("OriACC", row, float64(1))
	} else {
		dt.SetCellFloat("dOri", row, math.Round(float64(dec_ori*360)))
		dt.SetCellFloat("OriErr", row, float64(int(math.Abs(math.Round(float64(dec_ori*360))-float64(env.Angle)))%180))
		dt.SetCellFloat("OriACC", row, float64(0))
	}
	dt.SetCellString("ActAction", row, ss.ActAction)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
	//dt.SetCellString("TrialName", row, ss.TrainEnv.TrialName.Cur)
	for i, lnm := range ss.TargetLays {
		dt.SetCellFloat(lnm+"_CosDiff", row, float64(ss.TrlCosDiffTGT[i]))
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TrnTrlPlot != nil {
		ss.TrnTrlPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	//nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"Event", etensor.INT64, nil, nil},
		{"X", etensor.FLOAT64, nil, nil},
		{"Y", etensor.FLOAT64, nil, nil},
		{"dX", etensor.FLOAT64, nil, nil},
		{"dY", etensor.FLOAT64, nil, nil},
		{"PosErr", etensor.FLOAT64, nil, nil},
		{"PosACC", etensor.FLOAT64, nil, nil},
		{"Ori", etensor.FLOAT64, nil, nil},
		{"dOri", etensor.FLOAT64, nil, nil},
		{"OriErr", etensor.FLOAT64, nil, nil},
		{"OriACC", etensor.FLOAT64, nil, nil},
		{"ActAction", etensor.STRING, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}

	for _, lnm := range ss.TargetLays {
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "CAN_EC Train Trial Plot"
	plt.Params.XAxisCol = "Event"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, true, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, true, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, true, 0, eplot.FloatMax, 0)
	plt.SetColParams("Event", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Y", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("dX", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("dY", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PosErr", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PosACC", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Ori", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("dOri", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("OriErr", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("OriACC", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("ActAction", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.TargetLays {
		plt.SetColParams(lnm+"_CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}

	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	// nt := float64(ss.TrainEnv.Trial.Max)

	//ss.ECRFs()
	nt := float64(ss.NumTrlStats)
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	ss.NumTrlStats = 0

	trl := ss.TrnTrlLog
	trlix := etable.NewIdxView(trl)
	//gpsp := split.GroupBy(trlix, []string{"ActAction"})
	//for _, lnm := range ss.TargetLays {
	//	_, err := split.AggTry(gpsp, lnm+"_CosDiff", agg.AggMean)
	//	if err != nil {
	//		log.Println(err)
	//	}
	//}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

	for _, lnm := range ss.TargetLays {
		dt.SetCellFloat(lnm+"_CosDiff", row, agg.Agg(trlix, lnm+"_CosDiff", agg.AggMean)[0])
	}
	dt.SetCellFloat("PosErr", row, agg.Agg(trlix, "PosErr", agg.AggMean)[0])
	dt.SetCellFloat("PosACC", row, agg.Agg(trlix, "PosACC", agg.AggMean)[0])
	dt.SetCellFloat("OriErr", row, agg.Agg(trlix, "OriErr", agg.AggMean)[0])
	dt.SetCellFloat("OriACC", row, agg.Agg(trlix, "OriACC", agg.AggMean)[0])

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TargetLays {
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
	}
	sch = append(sch, etable.Column{"PosErr", etensor.FLOAT64, nil, nil})
	sch = append(sch, etable.Column{"PosACC", etensor.FLOAT64, nil, nil})
	sch = append(sch, etable.Column{"OriErr", etensor.FLOAT64, nil, nil})
	sch = append(sch, etable.Column{"OriACC", etensor.FLOAT64, nil, nil})

	dt.SetFromSchema(sch, 0)
	ss.ConfigWts(ss.EConWts)
	ss.ConfigWts(ss.ECoffWts)
	ss.ConfigWts(ss.ECWts)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "CAN_EC Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	for _, lnm := range ss.TargetLays {
		plt.SetColParams(lnm+"_CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	plt.SetColParams("PosErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PosACC", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("OriErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("OriACC", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	env := &ss.TrainEnv

	dt.SetCellFloat("Run", row, float64(env.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(env.Epoch.Cur))
	dt.SetCellFloat("Event", row, float64(env.Event.Cur))
	dt.SetCellFloat("X", row, float64(env.PosI.X))
	dt.SetCellFloat("Y", row, float64(env.PosI.Y))
	dt.SetCellFloat("Angle", row, float64(env.Angle))
	dt.SetCellString("ActAction", row, ss.ActAction)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	//epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	//
	//trl := ss.TestEnv.Trial.Cur
	//row := trl
	//
	//if dt.Rows <= row {
	//	dt.SetNumRows(row + 1)
	//}
	//
	//dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	//dt.SetCellFloat("Epoch", row, float64(epc))
	//dt.SetCellFloat("Trial", row, float64(trl))
	//dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	//nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Event", etensor.INT64, nil, nil},
		{"X", etensor.FLOAT64, nil, nil},
		{"Y", etensor.FLOAT64, nil, nil},
		{"Angle", etensor.FLOAT64, nil, nil},
		{"ActAction", etensor.STRING, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "EC Test Trial Plot"
	plt.Params.XAxisCol = "Event"
	plt.SetTable(dt)

	plt.SetColParams("Run", eplot.Off, true, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, true, 0, eplot.FloatMax, 0)
	plt.SetColParams("Event", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Angle", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("ActAction", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	// order of params: on, fixMin, min, fixMax, max 0)

	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	// trl := ss.TstTrlLog
	// tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "CAN_EC Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TrnEpcLog
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 10
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast-1:]

	// params := ss.Params.Name
	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)

	// runix := etable.NewIdxView(dt)
	// spl := split.GroupBy(runix, []string{"Params"})
	// split.Desc(spl, "FirstZero")
	// split.Desc(spl, "PctCor")
	// ss.RunStats = spl.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "CAN_EC Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	cam := &(nv.Scene().Camera)
	cam.Pose.Pos.Set(0.0, 1.08, 2.7)
	cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	// cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-1, 0, 0}, 0.4077744)
}

// ConfigWorldGui configures all the world view GUI elements
func (ss *Sim) ConfigWorldGui() *gi.Window {
	// order: Empty, wall, food, water, foodwas, waterwas
	ss.MatColors = []string{"lightgrey", "black", "orange", "blue", "brown", "navy"}

	ss.Trace = ss.TrainEnv.World.Clone().(*etensor.Int)
	ss.dTrace = ss.TrainEnv.World.Clone().(*etensor.Int)

	width := 1600
	height := 1200

	win := gi.NewMainWindow("xyhdenv", "XY and Head Direction Environment", width, height)
	ss.WorldWin = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(&ss.TrainEnv)

	tv := gi.AddNewTabView(split, "tv")
	ss.WorldTabs = tv

	tg := tv.AddNewTab(etview.KiT_TensorGrid, "Trace").(*etview.TensorGrid)
	ss.TraceView = tg
	tg.SetTensor(ss.Trace)
	ss.ConfigWorldView(tg)

	dtg := tv.AddNewTab(etview.KiT_TensorGrid, "dTrace").(*etview.TensorGrid)
	ss.dTraceView = dtg
	dtg.SetTensor(ss.dTrace)
	ss.ConfigWorldView(dtg)

	wg := tv.AddNewTab(etview.KiT_TensorGrid, "World").(*etview.TensorGrid)
	ss.WorldView = wg
	wg.SetTensor(ss.TrainEnv.World)
	ss.ConfigWorldView(wg)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "reset", Tooltip: "Init env.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetFullReRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Left", Icon: "wedge-left", Tooltip: "Rotate Left", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Left()
		vp.SetFullReRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Right", Icon: "wedge-right", Tooltip: "Rotate Right", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Right()
		vp.SetFullReRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Forward", Icon: "wedge-up", Tooltip: "Step Forward", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Forward()
		vp.SetFullReRender()
	})

	tbar.AddSeparator("sep-file")

	tbar.AddAction(gi.ActOpts{Label: "Open World", Icon: "file-open", Tooltip: "Open World from .tsv file", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(&ss.TrainEnv, "OpenWorld", vp)
	})

	tbar.AddAction(gi.ActOpts{Label: "Save World", Icon: "file-save", Tooltip: "Save World to .tsv file", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(&ss.TrainEnv, "SaveWorld", vp)
	})

	tbar.AddAction(gi.ActOpts{Label: "Open Pats", Icon: "file-open", Tooltip: "Open bit patterns from .json file", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(&ss.TrainEnv, "OpenPats", vp)
	})

	tbar.AddAction(gi.ActOpts{Label: "Save Pats", Icon: "file-save", Tooltip: "Save bit patterns to .json file", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(&ss.TrainEnv, "SavePats", vp)
	})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	win.MainMenuUpdated()
	return win
}

func (ss *Sim) ConfigWorldView(tg *etview.TensorGrid) {
	cnm := "XYHDEnvColors"
	cm, ok := giv.AvailColorMaps[cnm]
	if !ok {
		cm = &giv.ColorMap{}
		cm.Name = cnm
		cm.Indexed = true
		nc := len(ss.TrainEnv.Mats)
		cm.Colors = make([]gist.Color, nc+ss.TrainEnv.NRotAngles)
		cm.NoColor = gist.Black
		for i, cnm := range ss.MatColors {
			cm.Colors[i].SetString(cnm, nil)
		}
		ch := giv.AvailColorMaps["ColdHot"]
		for i := 0; i < ss.TrainEnv.NRotAngles; i++ {
			nv := float64(i) / float64(ss.TrainEnv.NRotAngles-1)
			cm.Colors[nc+i] = ch.Map(nv) // color map of rotation
		}
		giv.AvailColorMaps[cnm] = cm
	}
	tg.Disp.Defaults()
	tg.Disp.ColorMap = giv.ColorMapName(cnm)
	tg.Disp.GridFill = 1
	tg.SetStretchMax()
}

func (ss *Sim) UpdateWorldGui() {
	if ss.WorldWin == nil || !ss.TrainEnv.Disp {
		return
	}

	if ss.TrainEnv.Scene.Chg { // something important happened, refresh
		ss.Trace.CopyFrom(ss.TrainEnv.World)
		ss.dTrace.CopyFrom(ss.TrainEnv.World)
	}

	nc := len(ss.TrainEnv.Mats)
	ss.Trace.Set([]int{ss.TrainEnv.PosI.Y, ss.TrainEnv.PosI.X}, nc+ss.TrainEnv.Angle/ss.TrainEnv.AngInc)

	////////////////////////////////////// decoding trace
	env := &ss.TrainEnv
	pos := ss.Net.LayerByName("Out_Position").(leabra.LeabraLayer).AsLeabra()
	pos_tsr := &etensor.Float32{}
	pos_tsr.SetShape([]int{env.PosSize.Y, env.PosSize.X}, nil, []string{"Y", "X"})
	for i, val := range pos.Neurons {
		pos_tsr.Values[i] = val.ActM
	}
	dec_pos, _ := env.PopCode2d.Decode(pos_tsr)
	dX := int(math.Round(float64(dec_pos.X * (float32(env.Size.X) - 2))))
	dY := int(math.Round(float64(dec_pos.Y * (float32(env.Size.Y) - 2))))

	ori := ss.Net.LayerByName("Orientation").(leabra.LeabraLayer).AsLeabra()
	ori_tsr := make([]float32, len(ori.Neurons))
	for i, val := range ori.Neurons {
		ori_tsr[i] = val.ActM
	}
	dec_ori := env.AngCode.Decode(ori_tsr)
	dOri := int(math.Round(float64(dec_ori * 360)))

	ss.dTrace.Set([]int{dY, dX}, nc+dOri/env.AngInc)
	////////////////////////////////////////////////////////////////////////

	updt := ss.WorldTabs.UpdateStart()
	ss.TraceView.UpdateSig()
	ss.dTraceView.UpdateSig()
	ss.WorldTabs.UpdateEnd(updt)
}

func (ss *Sim) Left() {
	ss.TrainEnv.Action("Left", nil)
	ss.UpdateWorldGui()
}

func (ss *Sim) Right() {
	ss.TrainEnv.Action("Right", nil)
	ss.UpdateWorldGui()
}

func (ss *Sim) Forward() {
	ss.TrainEnv.Action("Forward", nil)
	ss.UpdateWorldGui()
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("can_ec")
	gi.SetAppAbout(`Alan testing out EC`)

	win := gi.NewMainWindow("can_ec", "EC Receptive Fields", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	//tg := tv.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	//tg.SetStretchMax()
	//ss.CurImgGrid = tg
	//tg.SetTensor(&ss.TrainEnv.Vis.ImgTsr)

	//tg = tv.AddNewTab(etview.KiT_TensorGrid, "EC RFs").(*etview.TensorGrid)
	//tg.SetStretchMax()
	//ss.WtsGrid = tg
	//tg.SetTensor(ss.ECWts) // zycyc comment out these two chunks

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("spec")

	//tbar.AddAction(gi.ActOpts{Label: "Open Rec=.2 Wts", Icon: "updt", Tooltip: "Open weights trained with excitatory lateral (recurrent) con scale = .2.", UpdateFunc: func(act *gi.Action) {
	//	act.SetActiveStateUpdt(!ss.IsRunning)
	//}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	//	ss.OpenRec2Wts()
	//})

	//tbar.AddAction(gi.ActOpts{Label: "Open Rec=.05 Wts", Icon: "updt", Tooltip: "Open weights trained with excitatory lateral (recurrent) con scale = .05.", UpdateFunc: func(act *gi.Action) {
	//	act.SetActiveStateUpdt(!ss.IsRunning)
	//}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	//	ss.OpenRec05Wts()
	//})

	tbar.AddAction(gi.ActOpts{Label: "Reset ARFs", Icon: "reset", Tooltip: "reset current position activation rfs accumulation data", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.ARFs.Reset()
	})

	tbar.AddAction(gi.ActOpts{Label: "View ARFs", Icon: "file-image", Tooltip: "compute activation rfs and view them.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.ARFs.Avg()
		ss.ARFs.Norm()
		for _, paf := range ss.ARFs.RFs {
			etview.TensorGridDialog(vp, &paf.NormRF, giv.DlgOpts{Title: "Act RF " + paf.Name, Prompt: paf.Name, TmpSave: nil}, nil, nil)
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Open ARFs", Icon: "file-open", Tooltip: "Open saved ARF .tsv files -- select a path or specific file in path", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(ss, "OpenAllARFs", vp)
	})

	//tbar.AddAction(gi.ActOpts{Label: "EC RFs", Icon: "file-image", Tooltip: "Update the EC Receptive Field (Weights) plot in EC RFs tab.", UpdateFunc: func(act *gi.Action) {
	//	act.SetActiveStateUpdt(!ss.IsRunning)
	//}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	//	ss.ECRFs()
	//})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't break on chg
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "update", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/ccnlab/map-nav/tree/master/sims/can_ec/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
		{"OpenAllARFs", ki.Props{
			"desc": "open all Activation-based Receptive Fields from selected path (can select a file too)",
			"icon": "file-open",
			"Args": ki.PropSlice{
				{"Path", ki.Props{
					"ext": ".tsv",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.MaxRuns, "runs", 1, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.SaveWts, "wts", true, "if true, save final weights after each run")
	flag.BoolVar(&ss.SaveARFs, "arfs", true, "if true, save final arfs after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.BoolVar(&ss.UseMPI, "mpi", false, "if set, use MPI for distributed computation")
	flag.Parse()
	ss.Init()

	//if ss.UseMPI {
	//	ss.MPIInit()
	//}

	// key for Config and Init to be after MPIInit
	ss.Config()
	ss.Init()

	if note != "" {
		mpi.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		mpi.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("trn_epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving training epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
		fnm = ss.LogFileName("tst_epc")
		ss.TstEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TstEpcFile = nil
		} else {
			fmt.Printf("Saving testing epoch log to: %v\n", fnm)
			defer ss.TstEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs\n", ss.MaxRuns)
	ss.Train()
}

////////////////////////////////////////////////////////////////////
//  MPI code

//// MPIInit initializes MPI
//func (ss *Sim) MPIInit() {
//	mpi.Init()
//	var err error
//	ss.Comm, err = mpi.NewComm(nil) // use all procs
//	if err != nil {
//		log.Println(err)
//		ss.UseMPI = false
//	} else {
//		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
//	}
//}
//
//// MPIFinalize finalizes MPI
//func (ss *Sim) MPIFinalize() {
//	if ss.UseMPI {
//		mpi.Finalize()
//	}
//}
//
//// CollectDWts collects the weight changes from all synapses into AllDWts
//func (ss *Sim) CollectDWts(net *axon.Network) {
//	net.CollectDWts(&ss.AllDWts)
//}
//
//// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
//// DWt changes across parallel nodes, each of which are learning on different
//// sequences of inputs.
//func (ss *Sim) MPIWtFmDWt() {
//	if ss.UseMPI {
//		ss.CollectDWts(&ss.Net.Network)
//		ndw := len(ss.AllDWts)
//		if len(ss.SumDWts) != ndw {
//			ss.SumDWts = make([]float32, ndw)
//		}
//		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
//		ss.Net.SetDWts(ss.SumDWts, mpi.WorldSize())
//	}
//	ss.Net.WtFmDWt()
//}
