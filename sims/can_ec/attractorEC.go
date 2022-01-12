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
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

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

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
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
					"Prjn.Learn.Norm.On":      "false",
					"Prjn.Learn.Momentum.On":  "false",
					"Prjn.Learn.WtBal.On":     "false",
					"Prjn.Learn.XCal.MLrn":    "0", // pure hebb
					"Prjn.Learn.XCal.SetLLrn": "true",
					"Prjn.Learn.XCal.LLrn":    "1",
					"Prjn.Learn.WtSig.Gain":   "1", // key: more graded weights
					"Prjn.Learn.Learn":        "false",
					"Prjn.WtInit.Mean":        "0.5",
					"Prjn.WtInit.Var":         "0.0", // even .01 causes some issues..
				}},
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					"Layer.Learn.AvgL.Gain":   "1", // this is critical! much lower
					"Layer.Learn.AvgL.Min":    "0.01",
					"Layer.Learn.AvgL.Init":   "0.2",
					"Layer.Inhib.Layer.Gi":    "1.8", // more active..
					"Layer.Inhib.Layer.FBTau": "3",
					"Layer.Inhib.ActAvg.Init": "0.2",
					"Layer.Act.Gbar.L":        "0.1",
					"Layer.Act.Dt.GTau":       "3", // slower = more noise integration -- otherwise fails sometimes
					"Layer.Act.Noise.Dist":    "Gaussian",
					"Layer.Act.Noise.Var":     "0.004", // 0.002 fails to converge sometimes, .005 a bit noisy
					"Layer.Act.Noise.Type":    "GeNoise",
					"Layer.Act.Noise.Fixed":   "false",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
				}},
			{Sel: ".ExciteLateral", Desc: "lateral excitatory connection",
				Params: params.Params{
					"Prjn.WtInit.Mean": "0.5",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
					"Prjn.WtScale.Rel": "0.2", // this controls the speed -- higher = faster
				}},
			{Sel: ".InhibLateral", Desc: "lateral inhibitory connection",
				Params: params.Params{
					"Prjn.WtInit.Mean": "0.5",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
					"Prjn.WtScale.Abs": "2", // higher gives better grid
				}},
			{Sel: ".OrientationForward", Desc: "orientation to ec forward connection",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.1",
				}},
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
	TrnTrlLog        *etable.Table    `view:"no-inline" desc:"training trial-level log data"`
	TrnEpcLog        *etable.Table    `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog        *etable.Table    `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog        *etable.Table    `view:"no-inline" desc:"testing trial-level log data"`
	RunLog           *etable.Table    `view:"no-inline" desc:"summary log of each run"`
	RunStats         *etable.Table    `view:"no-inline" desc:"aggregate stats on all runs"`
	Params           params.Sets      `view:"no-inline" desc:"full collection of param sets"`
	ParamSet         string           `view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	EConWts          *etensor.Float32 `view:"-" desc:"weights from input to EC layer"`
	ECoffWts         *etensor.Float32 `view:"-" desc:"weights from input to EC layer"`
	ECWts            *etensor.Float32 `view:"no-inline" desc:"net on - off weights from input to EC layer"`
	MaxRuns          int              `desc:"maximum number of model runs to perform"`
	MaxEpcs          int              `desc:"maximum number of epochs to run per model run"`
	//MaxTrls           int               `desc:"maximum number of training trials per epoch"`
	NZeroStop  int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv   env.FixedTable    `desc:"Training environment -- visual images"`
	TestEnv    env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time       leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn     bool              `desc:"whether to update the network view while running"`
	TrainUpdt  leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt   leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	LayStatNms []string          `desc:"names of layers to collect more detailed stats on (avg act, etc)"`

	// statistics: note use float64 as that is best for etable.Table
	Win         *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView     *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar     *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	CurImgGrid  *etview.TensorGrid          `view:"-" desc:"the current image grid view"`
	WtsGrid     *etview.TensorGrid          `view:"-" desc:"the weights grid view"`
	TrnTrlPlot  *eplot.Plot2D               `view:"-" desc:"the training trial plot"`
	TrnEpcPlot  *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot  *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot  *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	RunPlot     *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile  *os.File                    `view:"-" desc:"log file"`
	RunFile     *os.File                    `view:"-" desc:"log file"`
	ValsTsrs    map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	IsRunning   bool                        `view:"-" desc:"true if sim is running"`
	StopNow     bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed     int64                       `view:"-" desc:"the current random seed"`
}

// EcParams have the entorhinal cortex size and connectivity parameters
type EcParams struct {
	ECSize            evec.Vec2i `desc:"size of EC"`
	InputSize         evec.Vec2i `desc:"size of Input"`
	OrientationSize   evec.Vec2i `desc:"size of Input"`
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
	ss.LayStatNms = []string{"EC"}

	ss.Entorhinal.Defaults()
	ss.Pat.Defaults()
}

func (ec *EcParams) Defaults() {
	ec.ECSize.Set(50, 50)
	ec.InputSize.Set(2, 2)
	ec.OrientationSize.Set(2, 2)
	ec.InputPctAct = 0.25
	ec.OrientationPctAct = 0.25

	ec.excitRadius2D = 5
	ec.excitSigma2D = 3
	ec.inhibRadius2D = 10
	ec.inhibSigma2D = 10

	ec.excitRadius4D = 3
	ec.excitSigma4D = 2
	ec.inhibRadius4D = 10
	ec.inhibSigma4D = 2 // not really sure what this should be, seems like as long as it's not too small it's fine, 2 looks best
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
	ss.ConfigPats()
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
		ss.MaxEpcs = 30 // zycyc, test
		ss.NZeroStop = -1
	}
	//if ss.MaxTrls == 0 { // allow user override
	//	ss.MaxTrls = 100
	//}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	//ss.TrainEnv.Defaults()
	//ss.TrainEnv.ImageFiles = []string{"v1rf_img1.jpg", "v1rf_img2.jpg", "v1rf_img3.jpg", "v1rf_img4.jpg"}
	//ss.TrainEnv.OpenImagesAsset()
	ss.TrainEnv.Table = etable.NewIdxView(ss.OrientationInput)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	//ss.TrainEnv.Trial.Max = ss.MaxTrls

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing (probe) params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.Probes)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	ecParam := &ss.Entorhinal
	net.InitName(net, "attractorEC")
	input := net.AddLayer2D("Input", ecParam.InputSize.Y, ecParam.InputSize.X, emer.Input)
	orientation := net.AddLayer2D("Orientation", ecParam.OrientationSize.Y, ecParam.OrientationSize.X, emer.Input)
	ec := net.AddLayer4D("EC", ecParam.ECSize.Y, ecParam.ECSize.X, 2, 2, emer.Hidden) // 4D EC
	//ec := net.AddLayer2D("EC", ecParam.ECSize.Y, ecParam.ECSize.X, emer.Hidden) // 2D EC

	full := prjn.NewFull()
	net.ConnectLayers(input, ec, full, emer.Forward)

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

	oriePrjn := prjn.NewPoolSameUnit()
	orie := net.ConnectLayers(orientation, ec, oriePrjn, emer.Forward)
	orie.SetClass("OrientationForward")

	rec := net.ConnectLayers(ec, ec, excit, emer.Lateral)
	rec.SetClass("ExciteLateral")

	//inh := net.ConnectLayers(ec, ec, full, emer.Inhib)
	inh := net.ConnectLayers(ec, ec, inhib, emer.Inhib)
	inh.SetClass("InhibLateral")

	ec.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", XAlign: relpos.Left, YAlign: relpos.Front, Space: 2})
	orientation.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})

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
	ss.ConfigPats()
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
	ss.InitLateralWts(net)
}

func (ss *Sim) InitLateralWts(net *leabra.Network) {
	ecParam := &ss.Entorhinal
	ec := net.LayerByName("EC").(leabra.LeabraLayer).AsLeabra()
	lat := ec.RecvPrjn(2)
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
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
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

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()

	ec := ss.Net.LayerByName("EC").(leabra.LeabraLayer).AsLeabra()
	ec.Act.Clamp.Hard = false
	// Input pattern to stabilize grid sheet
	//for qtr := 0; qtr < 4; qtr++ {
	for qtr := 0; qtr < 10; qtr++ {
		//if qtr == 15 {
		//
		//	ss.Net.InitExt()
		//
		//
		//	ss.TrainEnv.Table = etable.NewIdxView(ss.OrientationInput)
		//
		//	// set names after updating epochs to get correct names for the next env
		//	ss.TrainEnv.SetTrialName()
		//	ss.TrainEnv.SetGroupName()
		//	ss.ApplyInputs(&ss.TrainEnv)
		//}
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
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
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

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Orientation"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
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

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}

		if epc >= ss.MaxEpcs {
			// done with training..
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
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Table = etable.NewIdxView(ss.OrientationInput)
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
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
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
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

// SaveWts saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWts(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
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

func (ss *Sim) ECRFs() {
	onVals := ss.EConWts.Values
	//offVals := ss.ECoffWts.Values
	netVals := ss.ECWts.Values
	on := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra() // zycyc: ??
	//off := ss.Net.LayerByName("LGNoff").(leabra.LeabraLayer).AsLeabra()
	isz := on.Shape().Len()
	ec := ss.Net.LayerByName("EC").(leabra.LeabraLayer).AsLeabra()
	ysz := ec.Shape().Dim(0)
	xsz := ec.Shape().Dim(1)
	for y := 0; y < ysz; y++ {
		for x := 0; x < xsz; x++ {
			ui := (y*xsz + x)
			ust := ui * isz
			onvls := onVals[ust : ust+isz]
			//offvls := offVals[ust : ust+isz]
			netvls := netVals[ust : ust+isz]
			on.SendPrjnVals(&onvls, "Wt", ec, ui, "")
			//off.SendPrjnVals(&offvls, "Wt", ec, ui, "")
			for ui := 0; ui < isz; ui++ {
				//netvls[ui] = 1.5 * (onvls[ui] - offvls[ui])
				netvls[ui] = 1.5 * (onvls[ui]) // zycyc: not sure what this is ??
			}
		}
	}
	if ss.WtsGrid != nil {
		ss.WtsGrid.UpdateSig()
	}
}

func (ss *Sim) ConfigWts(dt *etensor.Float32) {
	dt.SetShape([]int{14, 14, 12, 12}, nil, nil)
	dt.SetMetaData("grid-fill", "1")
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on chg, don't present
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

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

func (ss *Sim) ConfigPats() {
	ec := &ss.Entorhinal
	inputY := ec.InputSize.Y
	inputX := ec.InputSize.X
	orientationY := ec.OrientationSize.Y
	orientationX := ec.OrientationSize.X
	npats := ss.Pat.ListSize
	pctAct := ec.InputPctAct
	orientationPctAct := ec.OrientationPctAct
	//minDiff := ss.Pat.MinDiffPct

	patgen.AddVocabPermutedBinary(ss.PoolVocab, "randompats", npats, inputY, inputX, pctAct, 0)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "randomorientation", npats, orientationY, orientationX, orientationPctAct, 0)

	InitPatsSingle(ss.CorticalInput, "CorticalInput", "cortical input patterns to EC", []string{"Input"}, npats, []int{inputY}, []int{inputX})
	MixPats(ss.CorticalInput, ss.PoolVocab, []string{"Input"}, []string{"randompats"})

	InitPats(ss.OrientationInput, "OrientationInput", "only orientation input patterns to EC", []string{"Input", "Orientation"}, npats, []int{inputY, orientationY}, []int{inputX, orientationX})
	MixPats(ss.OrientationInput, ss.PoolVocab, []string{"Input", "Orientation"}, []string{"randompats", "randomorientation"})

}

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

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur

	row := dt.Rows
	if trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TrainEnv.TrialName.Cur)
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		tsr := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tsr, "Act")
		dt.SetCellTensor(lnm+"Act", row, tsr)
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

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm + "Act", etensor.FLOAT64, ly.Shp.Shp, nil})
	}

	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Train Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"Act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
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

	ss.ECRFs()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

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
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
	ss.ConfigWts(ss.EConWts)
	ss.ConfigWts(ss.ECoffWts)
	ss.ConfigWts(ss.ECWts)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "EC Receptive Field Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", eplot.On, eplot.FixMin, 0, eplot.FixMax, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TestEnv.Trial.Cur
	row := trl

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "EC Receptive Field Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
	}
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
	plt.Params.Title = "EC Receptive Field Testing Epoch Plot"
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
	params := "params"

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
	plt.Params.Title = "EC Receptive Field Run Plot"
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
	cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	// cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-1, 0, 0}, 0.4077744)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("attractorEC")
	gi.SetAppAbout(`Alan testing out EC`)

	win := gi.NewMainWindow("attractorEC", "EC Receptive Fields", width, height)
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

	tbar.AddAction(gi.ActOpts{Label: "EC RFs", Icon: "file-image", Tooltip: "Update the EC Receptive Field (Weights) plot in EC RFs tab.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.ECRFs()
	})

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

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
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
			gi.OpenURL("https://github.com/zycyc/attractorEC/README.md")
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
		{"SaveWts", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
