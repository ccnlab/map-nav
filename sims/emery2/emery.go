// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// emery2 is a simulated virtual rat / cat, using axon spiking model
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/actrf"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/empi"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/metric"
	"github.com/emer/etable/norm"
	"github.com/emer/etable/pca"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/gist"
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

// see params_def.go for default params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net              *deep.Network                 `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	PctCortex        float64                       `desc:"proportion of action driven by the cortex vs. hard-coded reflexive subcortical"`
	PctCortexMax     float64                       `desc:"maximum PctCortex, when running on the schedule"`
	ARFs             actrf.RFs                     `view:"no-inline" desc:"activation-based receptive fields"`
	TrnEpcLog        *etable.Table                 `view:"no-inline" desc:"training epoch-level log data"`
	TrnTrlLog        *etable.Table                 `view:"no-inline" desc:"training trial-level log data"`
	TrnTrlRepLog     *etable.Table                 `view:"no-inline" desc:"training trial-level reps log data"`
	TrnTrlRepLogAll  *etable.Table                 `view:"no-inline" desc:"training trial-level reps log data"`
	TrnErrStats      *etable.Table                 `view:"no-inline" desc:"stats on train trials where errors were made"`
	TrnAggStats      *etable.Table                 `view:"no-inline" desc:"stats on all train trials"`
	TstEpcLog        *etable.Table                 `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog        *etable.Table                 `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog        *etable.Table                 `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstCycLog        *etable.Table                 `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog           *etable.Table                 `view:"no-inline" desc:"summary log of each run"`
	RunStats         *etable.Table                 `view:"no-inline" desc:"aggregate stats on all runs"`
	MinusCycles      int                           `desc:"number of minus-phase cycles"`
	PlusCycles       int                           `desc:"number of plus-phase cycles"`
	AllLaysOnEpc     int                           `desc:"epoch to turn all layers on -- start with some toggled off (TE etc)"`
	ErrLrMod         axon.LrateMod                 `view:"inline" desc:"learning rate modulation as function of error"`
	Params           params.Sets                   `view:"no-inline" desc:"full collection of param sets"`
	ParamSet         string                        `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag              string                        `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Prjn4x4Skp2      *prjn.PoolTile                `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x4Skp2Recip *prjn.PoolTile                `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn4x3Skp2      *prjn.PoolTile                `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x3Skp2Recip *prjn.PoolTile                `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn3x3Skp1      *prjn.PoolTile                `view:"no-inline" desc:"feedforward 3x3 skip 1 topo prjn"`
	Prjn4x4Skp4      *prjn.PoolTile                `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn"`
	Prjn4x4Skp4Recip *prjn.PoolTile                `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn, recip"`
	MaxRuns          int                           `desc:"maximum number of model runs to perform"`
	MaxEpcs          int                           `desc:"maximum number of epochs to run per model run"`
	TestEpcs         int                           `desc:"number of epochs of testing to run, cumulative after MaxEpcs of training"`
	RepsInterval     int                           `desc:"how often to analyze the representations"`
	TrainEnv         FWorld                        `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time             axon.Time                     `desc:"axon timing parameters and state"`
	ViewOn           bool                          `desc:"whether to update the network view while running"`
	TrainUpdt        axon.TimeScales               `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt         axon.TimeScales               `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval     int                           `desc:"how often to run through all the test patterns, in terms of training epochs"`
	CosDifActs       []string                      `view:"-" desc:"actions to track CosDif performance by"`
	InitOffNms       []string                      `desc:"names of layers to turn off initially"`
	LayStatNms       []string                      `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	ARFLayers        []string                      `desc:"names of layers to compute position activation fields on"`
	SpikeRecLays     []string                      `desc:"names of layers to record spikes of during testing"`
	SpikeRasters     map[string]*etensor.Float32   `desc:"spike raster data for different layers"`
	SpikeRastGrids   map[string]*etview.TensorGrid `desc:"spike raster plots for different layers"`

	// statistics: note use float64 as that is best for etable.Table
	RFMaps        map[string]*etensor.Float32 `view:"no-inline" desc:"maps for plotting activation-based receptive fields"`
	PulvLays      []string                    `view:"-" desc:"pulvinar layers -- for stats"`
	HidLays       []string                    `view:"-" desc:"hidden layers: super and CT -- for hogging stats"`
	SuperLays     []string                    `view:"-" desc:"superficial layers"`
	InputLays     []string                    `view:"-" desc:"input layers"`
	NetAction     string                      `inactive:"+" desc:"action activated by the cortical network"`
	GenAction     string                      `inactive:"+" desc:"action generated by subcortical code"`
	ActAction     string                      `inactive:"+" desc:"actual action taken"`
	ActMatch      float64                     `inactive:"+" desc:"1 if net action matches gen action, 0 otherwise"`
	TrlCosDiff    float64                     `inactive:"+" desc:"current trial's overall cosine difference"`
	TrlCosDiffTRC []float64                   `inactive:"+" desc:"current trial's cosine difference for pulvinar (TRC) layers"`
	EpcActMatch   float64                     `inactive:"+" desc:"last epoch's average act match"`
	EpcCosDiff    float64                     `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	NumTrlStats   int                         `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumActMatch   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	PCA           pca.PCA                     `view:"-" desc:"pca obj"`

	// internal state - view:"-"
	Win          *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	WorldWin     *gi.Window                  `view:"-" desc:"FWorld GUI window"`
	WorldTabs    *gi.TabView                 `view:"-" desc:"FWorld TabView"`
	MatColors    []string                    `desc:"color strings in material order"`
	Trace        *etensor.Int                `view:"no-inline" desc:"trace of movement for visualization"`
	TraceView    *etview.TensorGrid          `desc:"view of the activity trace"`
	WorldView    *etview.TensorGrid          `desc:"view of the world"`
	TrnEpcPlot   *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TrnTrlPlot   *eplot.Plot2D               `view:"-" desc:"the training trial plot"`
	TstEpcPlot   *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D               `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File                    `view:"-" desc:"log file"`
	TstEpcFile   *os.File                    `view:"-" desc:"log file"`
	RunFile      *os.File                    `view:"-" desc:"log file"`
	PopVals      []float32                   `view:"-" desc:"tmp pop code values"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	SaveARFs     bool                        `view:"-" desc:"for command-line run only, auto-save receptive field data"`
	NoGui        bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	StopNow      bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64                       `view:"-" desc:"the current random seed"`
	UseMPI       bool                        `view:"-" desc:"if true, use MPI to distribute computation across nodes"`
	SaveProcLog  bool                        `view:"-" desc:"if true, save logs per processor"`
	Comm         *mpi.Comm                   `view:"-" desc:"mpi communicator"`
	AllDWts      []float32                   `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts      []float32                   `view:"-" desc:"buffer of MPI summed dwt weight changes"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &deep.Network{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnTrlRepLog = &etable.Table{}
	ss.TrnTrlRepLogAll = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}

	ss.Time.Defaults()
	ss.MinusCycles = 150
	ss.PlusCycles = 50
	ss.RepsInterval = 10
	ss.AllLaysOnEpc = 2

	ss.ErrLrMod.Defaults()
	ss.ErrLrMod.Base = 0.05 // 0.05 >= .01, .1 -- hard to tell
	ss.ErrLrMod.Range.Set(0.2, 0.8)
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = axon.AlphaCycle
	ss.TestUpdt = axon.GammaCycle
	ss.CosDifActs = []string{"Forward", "Left", "Right"}
	ss.InitOffNms = []string{"MSTdP", "cIPLCT", "cIPLP", "PCCCT"}
	ss.LayStatNms = []string{"MSTd", "MSTdCT", "SMA", "SMACT"}
	ss.ARFLayers = []string{"MSTd", "cIPL", "PCC", "SMA"}
	ss.SpikeRecLays = []string{"cIPL", "PCC", "SMA"}
	ss.Defaults()
	ss.NewPrjns()
}

// Defaults set default param values
func (ss *Sim) Defaults() {
	ss.PctCortexMax = 0.9 // 0.5 before
	ss.TestInterval = 50000
}

// NewPrjns creates new projections
func (ss *Sim) NewPrjns() {
	ss.Prjn4x4Skp2 = prjn.NewPoolTile()
	ss.Prjn4x4Skp2.Size.Set(4, 4)
	ss.Prjn4x4Skp2.Skip.Set(2, 2)
	ss.Prjn4x4Skp2.Start.Set(-1, -1)
	ss.Prjn4x4Skp2.TopoRange.Min = 0.5

	ss.Prjn4x4Skp2Recip = prjn.NewPoolTileRecip(ss.Prjn4x4Skp2)

	ss.Prjn4x3Skp2 = prjn.NewPoolTile()
	ss.Prjn4x3Skp2.Size.Set(3, 4)
	ss.Prjn4x3Skp2.Skip.Set(0, 2)
	ss.Prjn4x3Skp2.Start.Set(0, -1)
	ss.Prjn4x3Skp2.TopoRange.Min = 0.5

	ss.Prjn4x3Skp2Recip = prjn.NewPoolTileRecip(ss.Prjn4x3Skp2)

	ss.Prjn3x3Skp1 = prjn.NewPoolTile()
	ss.Prjn3x3Skp1.Size.Set(3, 1)
	ss.Prjn3x3Skp1.Skip.Set(1, 1)
	ss.Prjn3x3Skp1.Start.Set(-1, -1)

	ss.Prjn4x4Skp4 = prjn.NewPoolTile()
	ss.Prjn4x4Skp4.Size.Set(4, 1)
	ss.Prjn4x4Skp4.Skip.Set(4, 1)
	ss.Prjn4x4Skp4.Start.Set(0, 0)
	ss.Prjn4x4Skp4Recip = prjn.NewPoolTileRecip(ss.Prjn4x4Skp4)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnTrlRepLog(ss.TrnTrlRepLog)
	ss.ConfigTrnTrlRepLog(ss.TrnTrlRepLogAll)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 1
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 200
		ss.TestEpcs = 1000
	}

	ss.TrainEnv.Config(200) // 1000) // n trials per epoch
	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Run.Max = ss.MaxRuns
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

func (ss *Sim) ConfigNet(net *deep.Network) {
	net.InitName(net, "Emery")

	full := prjn.NewFull()
	sameu := prjn.NewPoolSameUnit()
	sameu.SelfCon = false
	p1to1 := prjn.NewPoolOneToOne()

	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1
	_ = rndcut

	var parprjn prjn.Pattern
	parprjn = full

	ev := &ss.TrainEnv
	fsz := 1 + 2*ev.FoveaSize
	// popsize = 12

	// input / output layers:
	v2wd, v2wdp := net.AddInputTRC4D("V2Wd", ev.DepthPools, ev.NFOVRays, ev.DepthSize/ev.DepthPools, 1)
	v2fd, v2fdp := net.AddInputTRC4D("V2Fd", ev.DepthPools, fsz, ev.DepthSize/ev.DepthPools, 1) // FovDepth
	v2wd.SetClass("Depth")
	v2wdp.SetClass("Depth")
	v2fd.SetClass("Depth")
	v2fdp.SetClass("Depth")

	v1f, v1fp := net.AddInputTRC4D("V1F", 1, fsz, ev.PatSize.Y, ev.PatSize.X) // Fovea
	v1f.SetClass("Fovea")
	v1fp.SetClass("Fovea")

	s1s, s1sp := net.AddInputTRC4D("S1S", 1, 4, 2, 1) // ProxSoma
	s1s.SetClass("S1S")
	s1sp.SetClass("S1S")

	s1v, s1vp := net.AddInputTRC4D("S1V", 1, 2, ev.PopSize, 1) // Vestibular
	s1v.SetClass("S1V")
	s1vp.SetClass("S1V")

	ins := net.AddLayer4D("Ins", 1, len(ev.Inters), ev.PopSize, 1, emer.Input) // Inters = Insula
	ins.SetClass("Ins")

	m1 := net.AddLayer2D("M1", 10, 10, emer.Hidden)
	vl := net.AddLayer2D("VL", ev.PatSize.Y, ev.PatSize.X, emer.Target)  // Action
	act := net.AddLayer2D("Act", ev.PatSize.Y, ev.PatSize.X, emer.Input) // Action

	m1p := net.AddTRCLayer2D("M1P", 10, 10)
	m1p.Driver = "M1"

	mstd, mstdct, mstdp := net.AddSuperCTTRC4D("MSTd", ev.DepthPools/2, ev.NFOVRays/2, 8, 8)
	mstdct.RecvPrjns().SendName(mstd.Name()).SetPattern(p1to1)                                // todo: try ss.Prjn3x3Skp1 orig: p1to1
	net.ConnectLayers(mstdct, v2wdp, ss.Prjn4x4Skp2Recip, emer.Forward).SetClass("CTToPulv2") // 3 is too high
	net.ConnectLayers(v2wdp, mstd, ss.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(v2wdp, mstdct, ss.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	// net.ConnectCtxtToCT(v2wd, mstdct, ss.Prjn4x4Skp2).SetClass("CTFmSuper")
	// net.ConnectCtxtToCT(mstdct, mstdct, parprjn).SetClass("CTSelf") // important!

	cipl, ciplct, ciplp := deep.AddSuperCTTRC4D(net.AsAxon(), "cIPL", 3, 3, 8, 8)
	ciplct.RecvPrjns().SendName(cipl.Name()).SetPattern(full)
	// net.ConnectLayers(ciplct, v2wdp, full, emer.Forward).SetClass("ToPulv1")
	// net.ConnectLayers(v2wdp, cipl, full, emer.Back).SetClass("FmPulv")
	// net.ConnectLayers(v2wdp, ciplct, full, emer.Back).SetClass("FmPulv")
	// net.ConnectCtxtToCT(ciplct, ciplct, parprjn).SetClass("CTSelf")

	pcc, pccct := deep.AddSuperCT4D(net.AsAxon(), "PCC", 2, 2, 7, 7)
	pccct.RecvPrjns().SendName(pcc.Name()).SetPattern(parprjn)
	// net.ConnectLayers(pccct, v2wdp, full, emer.Forward).SetClass("ToPulv1")
	// net.ConnectLayers(v2wdp, pcc, full, emer.Back).SetClass("FmPulv")
	// net.ConnectLayers(v2wdp, pccct, full, emer.Back).SetClass("FmPulv")
	// net.ConnectCtxtToCT(pccct, pccct, parprjn).SetClass("CTSelf")

	sma, smact := net.AddSuperCT2D("SMA", 10, 10)
	smact.RecvPrjns().SendName(sma.Name()).SetPattern(full)
	// net.ConnectCtxtToCT(smact, smact, parprjn).SetClass("CTSelf")
	net.ConnectLayers(smact, m1p, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(m1p, sma, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(m1p, smact, full, emer.Back).SetClass("FmPulv")

	it, itct := net.AddSuperCT2D("IT", 10, 10)
	itct.RecvPrjns().SendName(it.Name()).SetPattern(parprjn)
	net.ConnectLayers(itct, v1fp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(v1fp, itct, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(v1fp, it, full, emer.Back).SetClass("FmPulv")
	// net.ConnectCtxtToCT(itct, itct, p1to1).SetClass("CTSelf")

	lip, lipct := net.AddSuperCT4D("LIP", ev.DepthPools/2, 1, 8, 8)
	lipct.RecvPrjns().SendName(lip.Name()).SetPattern(full)
	net.ConnectLayers(lipct, v2fdp, ss.Prjn4x3Skp2Recip, emer.Forward).SetClass("CTToPulv3")
	net.ConnectLayers(v2fdp, lipct, ss.Prjn4x3Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(v2fdp, lip, ss.Prjn4x3Skp2, emer.Back).SetClass("FmPulv")
	// net.ConnectCtxtToCT(lipct, lipct, full).SetClass("CTSelf")

	// todo: LIP fovea is not topo organized for left, middle right positions -- groups are depth organized
	// not enough resolution to really map that out here.
	net.ConnectLayers(lipct, v1fp, full, emer.Back).SetClass("ToPulv1") // attention

	m1.SetClass("M1")
	vl.SetClass("M1")
	act.SetClass("M1")
	m1p.SetClass("M1")

	mstd.SetClass("MSTd")
	mstdct.SetClass("MSTd")
	mstdp.SetClass("MSTd")

	cipl.SetClass("cIPL")
	ciplct.SetClass("cIPL")
	ciplp.SetClass("cIPL")

	pcc.SetClass("PCC")
	pccct.SetClass("PCC")

	sma.SetClass("SMA")
	smact.SetClass("SMA")

	it.SetClass("IT")
	itct.SetClass("IT")

	lip.SetClass("LIP")
	lipct.SetClass("LIP")

	////////////////////
	// basic super cons

	net.ConnectLayers(v2wd, mstd, ss.Prjn4x4Skp2, emer.Forward).SetClass("SuperFwd")

	// MStd <-> CIPl
	net.ConnectLayers(mstd, cipl, ss.Prjn4x4Skp2, emer.Forward).SetClass("SuperFwd")
	net.ConnectLayers(cipl, mstd, ss.Prjn4x4Skp2Recip, emer.Back)
	net.ConnectLayers(ciplct, mstdct, ss.Prjn4x4Skp2Recip, emer.Back).SetClass("CTBack")

	net.ConnectLayers(mstdp, ciplct, ss.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(mstdp, cipl, ss.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(ciplct, mstdp, ss.Prjn4x4Skp2Recip, emer.Forward).SetClass("CTToPulv3")

	net.ConnectLayers(smact, mstdct, full, emer.Back).SetClass("CTBack")
	net.ConnectLayers(sma, mstd, full, emer.Back)

	// CIPl <-> PCC
	net.ConnectLayers(cipl, pcc, parprjn, emer.Forward).SetClass("SuperFwd")
	net.ConnectLayers(pcc, cipl, parprjn, emer.Back)
	net.ConnectLayers(pccct, ciplct, parprjn, emer.Back).SetClass("CTBack")

	net.ConnectLayers(ciplp, pccct, parprjn, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(ciplp, pcc, parprjn, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(pccct, ciplp, parprjn, emer.Forward).SetClass("CTToPulv2")

	// PCC <-> SMA
	net.ConnectLayers(pcc, sma, parprjn, emer.Forward).SetClass("SuperFwd")
	net.ConnectLayers(sma, pcc, parprjn, emer.Back)
	net.ConnectLayers(smact, pccct, parprjn, emer.Back).SetClass("CTBack")

	// SMA <-> M1
	net.ConnectLayers(sma, m1, parprjn, emer.Forward).SetClass("SuperFwd")

	net.BidirConnectLayers(m1, vl, full)

	net.ConnectLayers(v1f, it, full, emer.Forward)
	net.ConnectLayers(v2fd, lip, ss.Prjn4x3Skp2, emer.Forward)

	////////////////////
	// to MSTd

	// net.ConnectLayers(mstd, mstd, sameu, emer.Lateral)

	// net.ConnectLayers(sma, mstd, parprjn, emer.Back)
	// net.ConnectLayers(s1v, mstd, full, emer.Back)

	// MSTdCT top-down depth
	// net.ConnectLayers(ciplct, mstdct, parprjn, emer.Back).SetClass("CTBack")
	// net.ConnectLayers(pccct, mstdct, parprjn, emer.Back).SetClass("CTBack")
	// net.ConnectLayers(smact, mstdct, parprjn, emer.Back).SetClass("CTBack") // always need sma to predict action outcome

	// ActToCT are used temporarily to endure prediction is properly contextualized
	// net.ConnectCtxtToCT(act, mstdct, full).SetClass("ActToCT")
	// net.ConnectCtxtToCT(act, ciplct, full).SetClass("ActToCT")
	// net.ConnectCtxtToCT(act, smact, full).SetClass("ActToCT")
	// net.ConnectCtxtToCT(act, pccct, full).SetClass("ActToCT")
	// net.ConnectCtxtToCT(act, itct, full).SetClass("ActToCT")
	// net.ConnectCtxtToCT(act, lipct, full).SetClass("ActToCT")

	net.ConnectCtxtToCT(m1p, mstdct, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, ciplct, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, smact, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, pccct, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, itct, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, lipct, full).SetClass("FmPulv")

	// // S1 vestibular
	// nt.ConnectLayers(mstdct, s1vp, full, emer.Forward)
	// nt.ConnectLayers(s1vp, mstdct, full, emer.Back).SetClass("FmPulv")
	// nt.ConnectLayers(s1vp, mstd, full, emer.Back).SetClass("FmPulv")

	// todo: try S -> CT leak back -- useful in wwi3d
	// todo: try higher CT -> v2wdp  -- useful in wwi3d

	////////////////////
	// to cIPL

	// net.ConnectLayers(sma, cipl, parprjn, emer.Back)
	net.ConnectLayers(s1s, cipl, full, emer.Back)
	net.ConnectLayers(s1v, cipl, full, emer.Back)
	// net.ConnectLayers(vl, cipl, full, emer.Back) // todo: m1?

	net.ConnectLayers(pccct, ciplct, parprjn, emer.Back).SetClass("CTBack")
	net.ConnectLayers(smact, ciplct, parprjn, emer.Back).SetClass("CTBack")

	// net.ConnectLayers(mstdct, ciplp, ss.Prjn4x4Skp2, emer.Forward).SetClass("FwdToPulv")

	// todo: try S -> CT leak back -- useful in wwi3d
	// todo: try higher CT -> v2wdp  -- useful in wwi3d

	// todo: S1Vp should be head direction too.

	// S1 vestibular
	net.ConnectLayers(ciplct, s1vp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1vp, ciplct, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1vp, cipl, full, emer.Back).SetClass("FmPulv")

	////////////////////
	// to PCC

	net.ConnectLayers(s1s, pcc, full, emer.Forward)
	net.ConnectLayers(s1v, pcc, full, emer.Forward)
	// net.ConnectLayers(vl, pcc, full, emer.Back)

	net.ConnectLayers(smact, pccct, parprjn, emer.Back).SetClass("CTBack")

	// S1 vestibular
	net.ConnectLayers(pccct, s1vp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1vp, pccct, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1vp, pcc, full, emer.Back).SetClass("FmPulv")

	// S1 soma
	net.ConnectLayers(pccct, s1sp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1sp, pccct, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1sp, pcc, full, emer.Back).SetClass("FmPulv")

	////////////////////
	// to SMA

	net.ConnectLayers(it, sma, full, emer.Forward)
	net.ConnectLayers(lip, sma, full, emer.Forward)
	net.ConnectLayers(s1s, sma, full, emer.Forward)

	// net.ConnectLayers(vl, sma, full, emer.Back)
	// net.ConnectLayers(cipl, sma, parprjn, emer.Forward) // todo: forward??
	// net.ConnectLayers(vl, smact, full, emer.Back)

	// S1 vestibular
	net.ConnectLayers(smact, s1vp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1vp, smact, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1vp, sma, full, emer.Back).SetClass("FmPulv")

	// S1 soma
	net.ConnectLayers(smact, s1sp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1sp, smact, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1sp, sma, full, emer.Back).SetClass("FmPulv")

	////////////////////
	// to M1

	net.ConnectLayers(smact, vl, full, emer.Forward)
	// net.ConnectLayers(sma, vl, full, emer.Forward) // no, right?

	////////////////////
	// to IT

	net.ConnectLayers(sma, it, full, emer.Back)
	// net.ConnectLayers(pcc, it, full, emer.Back) // not useful

	net.ConnectLayers(smact, itct, parprjn, emer.Back).SetClass("CTBack") // needs to know how moving..
	// net.ConnectLayers(pccct, itct, full, emer.Back).SetClass("CTBack")

	////////////////////
	// to LIP

	net.ConnectLayers(sma, lip, full, emer.Back)
	// net.ConnectLayers(pcc, lip, full, emer.Back) // not useful

	net.ConnectLayers(smact, lipct, full, emer.Back).SetClass("CTBack") // always need sma to predict action outcome
	// net.ConnectLayers(pccct, lipct, full, emer.Back).SetClass("CTBack")

	////////////////////
	// lateral inhibition

	net.LateralConnectLayerPrjn(mstd, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(mstdct, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(cipl, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(ciplct, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(pcc, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(pccct, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(sma, full, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(smact, full, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(m1, full, &axon.HebbPrjn{}).SetType(emer.Inhib)

	////////////////////
	// Shortcuts

	net.ConnectLayers(v2wd, cipl, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, ciplct, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, pcc, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, pccct, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, sma, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, smact, rndcut, emer.Forward).SetClass("V1SC")

	//////////////////////////////////////
	// position

	v1f.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v2wd.Name(), YAlign: relpos.Front, Space: 15})
	v2fd.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1f.Name(), XAlign: relpos.Left, Space: 4})
	v2wdp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v2wd.Name(), XAlign: relpos.Left, Space: 4})

	v1fp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1f.Name(), XAlign: relpos.Left, Space: 4})
	it.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1fp.Name(), XAlign: relpos.Left, Space: 4})
	itct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: it.Name(), XAlign: relpos.Left, Space: 4})

	v2fd.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1f.Name(), YAlign: relpos.Front, Space: 8})
	v2fdp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v2fd.Name(), XAlign: relpos.Left, Space: 10})
	lip.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v2fdp.Name(), YAlign: relpos.Front, Space: 20})
	lipct.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: lip.Name(), YAlign: relpos.Front, Space: 8})

	s1s.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v2fd.Name(), YAlign: relpos.Front, Space: 20})
	s1sp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: s1s.Name(), XAlign: relpos.Left, Space: 4})
	s1v.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: s1s.Name(), YAlign: relpos.Front, Space: 15})
	s1vp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: s1v.Name(), XAlign: relpos.Left, Space: 4})

	ins.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: s1v.Name(), YAlign: relpos.Front, Space: 20})

	vl.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: ins.Name(), YAlign: relpos.Front, Space: 10})
	act.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vl.Name(), XAlign: relpos.Left, Space: 4})

	mstd.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: v2wd.Name(), XAlign: relpos.Left, YAlign: relpos.Front})
	mstdct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: mstd.Name(), XAlign: relpos.Left, Space: 10})
	mstdp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: mstdct.Name(), XAlign: relpos.Left, Space: 10})

	cipl.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mstd.Name(), YAlign: relpos.Front, Space: 4})
	ciplct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: cipl.Name(), XAlign: relpos.Left, Space: 10})
	ciplp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: ciplct.Name(), XAlign: relpos.Left, Space: 10})

	pcc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: cipl.Name(), YAlign: relpos.Front, Space: 6})
	pccct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: pcc.Name(), XAlign: relpos.Left, Space: 10})
	// pccp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: pccct.Name(), XAlign: relpos.Left, Space: 4})

	sma.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: pcc.Name(), YAlign: relpos.Front, Space: 10})
	smact.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: sma.Name(), XAlign: relpos.Left, Space: 10})
	// smap.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "SMACT", XAlign: relpos.Left, Space: 4})

	m1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: sma.Name(), YAlign: relpos.Front, Space: 10})
	m1p.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: m1.Name(), XAlign: relpos.Left, Space: 10})

	//////////////////////////////////////
	// collect

	ss.PulvLays = make([]string, 0, 10)
	ss.HidLays = make([]string, 0, 10)
	ss.SuperLays = make([]string, 0, 10)
	ss.InputLays = make([]string, 0, 10)
	for _, ly := range net.Layers {
		if ly.IsOff() {
			continue
		}
		switch ly.Type() {
		case emer.Input:
			ss.InputLays = append(ss.InputLays, ly.Name())
		case deep.TRC:
			ss.PulvLays = append(ss.PulvLays, ly.Name())
		case emer.Hidden:
			ss.SuperLays = append(ss.SuperLays, ly.Name())
			fallthrough
		case deep.CT:
			ss.HidLays = append(ss.HidLays, ly.Name())
		}
	}
	ss.PulvLays = append(ss.PulvLays, "VL")

	// using 4 total threads -- todo: didn't work
	/*
		mstd.SetThread(1)
		mstdct.SetThread(1)
		cipl.SetThread(2)
		ciplct.SetThread(2)
		pcc.SetThread(3)
		pccct.SetThread(3)
		sma.SetThread(3)
		smact.SetThread(3)
	*/

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	if !ss.NoGui {
		sr := net.SizeReport()
		mpi.Printf("%s", sr)
	}
	ss.InitWts(net)
}

// Initialize network weights including scales
func (ss *Sim) InitWts(net *deep.Network) {
	net.InitWts()
	// net.InitTopoSWts() //  sets all wt scales

	// ss.ToggleLaysOff(true)
}

func (ss *Sim) ToggleLaysOff(off bool) {
	for _, lnm := range ss.InitOffNms {
		lyi := ss.Net.LayerByName(lnm)
		if lyi == nil {
			fmt.Printf("layer not found: %s\n", lnm)
			continue
		}
		lyi.SetOff(off)
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
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
	// if train {
	return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tEvent:\t%d\tCycle:\t%d\tAct:\t%v\tNet:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Event.Cur, ss.Time.Cycle, ss.ActAction, ss.NetAction)
	// } else {
	// 	return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tEvent:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Event.Cur, ss.Time.Cycle, ss.TrainEnv.Event.Cur)
	// }
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
	ss.UpdateWorldGui()
}

func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
	case axon.Cycle:
		ss.UpdateView(train)
	case axon.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.UpdateView(train)
		}
	case axon.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.UpdateView(train)
		}
	case axon.AlphaCycle:
		if ss.Time.Cycle%100 == 0 {
			ss.UpdateView(train)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// ThetaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of ThetaCycle
func (ss *Sim) ThetaCyc(train bool) {
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
	} else {
		ss.Net.SynFail()
	}

	ev := &ss.TrainEnv

	minusCyc := ss.MinusCycles
	plusCyc := ss.PlusCycles

	ss.Net.NewState()
	ss.Time.NewState()

	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)
		ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
		if !ss.NoGui {
			ss.RecordSpikes(ss.Time.Cycle)
		}
		ss.Time.CycleInc()
		switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
		case 75:
			ss.Net.ActSt1(&ss.Time)
			// if erand.BoolProb(float64(ss.PAlphaPlus), -1) {
			// 	ss.Net.TargToExt()
			// 	ss.Time.PlusPhase = true
			// }
		case 100:
			ss.Net.ActSt2(&ss.Time)
			ss.Net.ClearTargExt()
			ss.Time.PlusPhase = false
		}

		if cyc == minusCyc-1 { // do before view update
			ss.Net.MinusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.Time.NewPhase()
	ss.TakeAction(ss.Net, ev)
	if viewUpdt == axon.Phase {
		ss.UpdateView(train)
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)
		ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
		if !ss.NoGui {
			ss.RecordSpikes(ss.Time.Cycle)
		}
		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}

	ss.TrialStats(train) // need stats for lrmod

	if train {
		ss.Net.DWt()
	}
	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.UpdateView(train)
	}
	if !ss.NoGui {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) TakeAction(net *deep.Network, ev *FWorld) {
	ly := net.LayerByName("VL").(axon.AxonLayer).AsAxon()
	nact := ss.DecodeAct(ly, ev)
	gact, urgency := ev.ActGen()
	ss.NetAction = ev.Acts[nact]
	ss.GenAction = ev.Acts[gact]
	ss.ActMatch = 0
	if nact == gact {
		ss.ActMatch = 1
	}
	if erand.BoolProb(float64(urgency), -1) {
		ss.ActAction = ss.GenAction
	} else if erand.BoolProb(ss.PctCortex, -1) {
		ss.ActAction = ss.NetAction
	} else {
		ss.ActAction = ss.GenAction
	}
	ly.SetType(emer.Input)
	ev.Action(ss.ActAction, nil)
	ap, ok := ev.Pats[ss.ActAction]
	if ok {
		ly.ApplyExt(ap)
	}
	ly.SetType(emer.Target)
	// fmt.Printf("action: %s\n", ev.Acts[act])
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ly *axon.Layer, ev *FWorld) int {
	vt := ss.ValsTsr("VL")
	ly.UnitValsTensor(vt, "ActM")
	act := ev.DecodeAct(vt)
	return act
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(net *deep.Network, en env.Env) {
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "Vestibular", "Inters", "Action", "Action"}
	lays := []string{"V2Wd", "V2Fd", "V1F", "S1S", "S1V", "Ins", "VL", "Act"}
	for i, lnm := range lays {
		lyi := ss.Net.LayerByName(lnm)
		if lyi == nil {
			continue
		}
		ly := lyi.(axon.AxonLayer).AsAxon()
		pats := en.State(states[i])
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
		ss.EpochSched(epc)
		ss.TrainEnv.Event.Cur = 0
		if ss.ViewOn && ss.TrainUpdt > axon.ThetaCycle {
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

	ss.ApplyInputs(ss.Net, &ss.TrainEnv)
	ss.ThetaCyc(true) // train
	// ss.TrialStats(true) // now in alphacyc
	ss.LogTrnTrl(ss.TrnTrlLog)
	if ss.RepsInterval > 0 && epc%ss.RepsInterval == 0 {
		ss.LogTrnRepTrl(ss.TrnTrlRepLog)
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	// if ss.SaveWts { // doing this earlier
	// 	ss.SaveWeights()
	// }
	if ss.SaveARFs {
		ss.SaveAllARFs()
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.PctCortex = 0
	ss.TrainEnv.Init(run)
	// ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.NumTrlStats = 0
	ss.SumActMatch = 0
	ss.SumCosDiff = 0
	// clear rest just to make Sim look initialized
	ss.EpcActMatch = 0
	ss.EpcCosDiff = 0
}

// TrialStatsTRC computes the trial-level statistics for TRC layers
func (ss *Sim) TrialStatsTRC(accum bool) {
	nt := len(ss.PulvLays)
	if len(ss.TrlCosDiffTRC) != nt {
		ss.TrlCosDiffTRC = make([]float64, nt)
	}
	acd := 0.0
	for i, ln := range ss.PulvLays {
		ly := ss.Net.LayerByName(ln).(axon.AxonLayer).AsAxon()
		cd := float64(ly.CosDiff.Cos)
		acd += cd
		ss.TrlCosDiffTRC[i] = cd
	}
	ss.TrlCosDiff = acd / float64(len(ss.PulvLays))
	if accum {
		ss.SumCosDiff += ss.TrlCosDiff
	}
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
			mt.Set1D(ss.TrainEnv.Angle/15, 1)
		case "Rot":
			mt.Set1D(1+ss.TrainEnv.RotAng/15, 1)
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

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	ss.TrialStatsTRC(accum)
	if accum {
		ss.SumActMatch += ss.ActMatch
		ss.NumTrlStats++
	} else {
		ss.UpdtARFs() // only in testing
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

// EpochSched implements the learning rate schedule etc.
func (ss *Sim) EpochSched(epc int) {
	if epc > 1 && epc%10 == 0 {
		ss.PctCortex = float64(epc) / 100
		if ss.PctCortex > ss.PctCortexMax {
			ss.PctCortex = ss.PctCortexMax
		} else {
			fmt.Printf("PctCortex updated to: %g at epoch: %d\n", ss.PctCortex, epc)
		}
	}
	switch epc {
	// case ss.AllLaysOnEpc:
	// 	ss.ToggleLaysOff(false) // on
	// 	mpi.Printf("toggled layers on at epoch: %d\n", epc)
	case 150:
		ss.Net.LrateSched(0.5)
		fmt.Printf("dropped lrate 0.5 at epoch: %d\n", epc)
	case 250:
		ss.Net.LrateSched(0.2)
		fmt.Printf("dropped lrate 0.2 at epoch: %d\n", epc)
	case 350:
		ss.Net.LrateSched(0.1)
		fmt.Printf("dropped lrate 0.1 at epoch: %d\n", epc)
	}
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
		vp.BlockUpdates()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.UnblockUpdates()
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

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// note: using TrainEnv for everything

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTstEpc(ss.TstEpcLog)
		ss.EpochSched(epc)
		ss.TrainEnv.Event.Cur = 0
		if ss.ViewOn && ss.TrainUpdt > axon.ThetaCycle {
			ss.UpdateView(true)
		}
		if epc >= ss.TestEpcs {
			ss.StopNow = true
			return
		}
	}

	ss.ApplyInputs(ss.Net, &ss.TrainEnv)
	ss.ThetaCyc(false) // train
	// ss.TrialStats(true) // now in alphacyc
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
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
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
	}
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

// ConfigSpikeRasts
func (ss *Sim) ConfigSpikeRasts() {
	ncy := ss.MinusCycles + ss.PlusCycles
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		sr := ss.SpikeRastTsr(lnm)
		sr.SetShape([]int{ly.Shp.Len(), ncy}, nil, []string{"Nrn", "Cyc"})
	}
}

// SpikeRastTsr gets spike raster tensor of given name, creating if not yet made
func (ss *Sim) SpikeRastTsr(name string) *etensor.Float32 {
	if ss.SpikeRasters == nil {
		ss.SpikeRasters = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.SpikeRasters[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.SpikeRasters[name] = tsr
	}
	return tsr
}

// SpikeRastGrid gets spike raster grid of given name, creating if not yet made
func (ss *Sim) SpikeRastGrid(name string) *etview.TensorGrid {
	if ss.SpikeRastGrids == nil {
		ss.SpikeRastGrids = make(map[string]*etview.TensorGrid)
	}
	tsr, ok := ss.SpikeRastGrids[name]
	if !ok {
		tsr = &etview.TensorGrid{}
		ss.SpikeRastGrids[name] = tsr
	}
	return tsr
}

// SetSpikeRastCol sets column of given spike raster from data
func (ss *Sim) SetSpikeRastCol(sr, vl *etensor.Float32, col int) {
	for ni, v := range vl.Values {
		sr.Set([]int{ni, col}, v)
	}
}

// ConfigSpikeGrid configures the spike grid
func (ss *Sim) ConfigSpikeGrid(tg *etview.TensorGrid, sr *etensor.Float32) {
	tg.SetStretchMax()
	sr.SetMetaData("grid-fill", "1")
	tg.SetTensor(sr)
}

// RecordSpikes
func (ss *Sim) RecordSpikes(cyc int) {
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		tv := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tv, "Spike")
		sr := ss.SpikeRastTsr(lnm)
		ss.SetSpikeRastCol(sr, tv, cyc)
	}
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
//  TrnTrlRepLog

// CenterPoolsIdxs returns the indexes for 2x2 center pools (including sub-pools):
// nu = number of units per pool, sis = starting indexes
func (ss *Sim) CenterPoolsIdxs(ly *axon.Layer) (nu int, sis []int) {
	nu = ly.Shp.Dim(2) * ly.Shp.Dim(3)
	npy := ly.Shp.Dim(0)
	npx := ly.Shp.Dim(1)
	npxact := npx
	nsp := 1
	// if ss.SubPools {
	// 	npy /= 2
	// 	npx /= 2
	// 	nsp = 2
	// }
	cpy := (npy - 1) / 2
	cpx := (npx - 1) / 2
	if npx <= 2 {
		cpx = 0
	}
	if npy <= 2 {
		cpy = 0
	}

	for py := 0; py < 2; py++ {
		for px := 0; px < 2; px++ {
			for sy := 0; sy < nsp; sy++ {
				for sx := 0; sx < nsp; sx++ {
					y := (py+cpy)*nsp + sy
					x := (px+cpx)*nsp + sx
					si := (y*npxact + x) * nu
					sis = append(sis, si)
				}
			}
		}
	}
	return
}

// CopyCenterPools copy 2 center pools of ActM to tensor
func (ss *Sim) CopyCenterPools(ly *axon.Layer, vl *etensor.Float32) {
	nu, sis := ss.CenterPoolsIdxs(ly)
	vl.SetShape([]int{len(sis) * nu}, nil, nil)
	ti := 0
	for _, si := range sis {
		for ni := 0; ni < nu; ni++ {
			vl.Values[ti] = ly.Neurons[si+ni].ActM
			ti++
		}
	}
}

// LogTrnRepTrl adds data from current trial to the TrnTrlRepLog table.
func (ss *Sim) LogTrnRepTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	event := ss.TrainEnv.Event.Cur
	row := dt.Rows

	if row > 1 { // reset at new epoch
		lstepc := int(dt.CellFloat("Epoch", row-1))
		if lstepc != epc {
			dt.SetNumRows(0)
			row = 0
		}
	}
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Event", row, float64(event))
	dt.SetCellFloat("Idx", row, float64(row))
	dt.SetCellFloat("X", row, float64(ss.TrainEnv.PosI.X))
	dt.SetCellFloat("Y", row, float64(ss.TrainEnv.PosI.Y))
	dt.SetCellString("NetAction", row, ss.NetAction)
	dt.SetCellString("GenAction", row, ss.GenAction)
	dt.SetCellString("ActAction", row, ss.ActAction)
	dt.SetCellFloat("ActMatch", row, ss.ActMatch)

	for _, lnm := range ss.HidLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		lvt := ss.ValsTsr(lnm)
		if ly.Is4D() && ly.Shp.Dim(0) > 2 && ly.Shp.Dim(2) > 2 {
			ss.CopyCenterPools(ly, lvt)
			dt.SetCellTensor(lnm, row, lvt)
		} else {
			ly.UnitValsTensor(lvt, "ActM")
			dt.SetCellTensor(lnm, row, lvt)
		}
	}
}

func (ss *Sim) ConfigTrnTrlRepLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnTrlRepLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Event", etensor.INT64, nil, nil},
		{"X", etensor.FLOAT64, nil, nil},
		{"Y", etensor.FLOAT64, nil, nil},
		{"NetAction", etensor.STRING, nil, nil},
		{"GenAction", etensor.STRING, nil, nil},
		{"ActAction", etensor.STRING, nil, nil},
		{"ActMatch", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.HidLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		if ly.Is4D() && ly.Shp.Dim(0) > 2 && ly.Shp.Dim(2) > 2 {
			nu, sis := ss.CenterPoolsIdxs(ly)
			sch = append(sch, etable.Column{lnm, etensor.FLOAT64, []int{len(sis) * nu}, nil})
		} else {
			sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

//////////////////////////////////////////////
//  TrnEpcLog

// HogDead computes the proportion of units in given layer name with ActAvg over hog thr
// and under dead threshold
func (ss *Sim) HogDead(lnm string) (hog, dead float64) {
	ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	n := len(ly.Neurons)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.ActAvg > 0.3 {
			hog += 1
		} else if nrn.ActAvg < 0.01 {
			dead += 1
		}
	}
	hog /= float64(n)
	dead /= float64(n)
	return
}

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	nt := float64(ss.NumTrlStats)

	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	ss.EpcActMatch = ss.SumActMatch / nt
	ss.SumActMatch = 0

	ss.NumTrlStats = 0

	trl := ss.TrnTrlLog
	trlix := etable.NewIdxView(trl)
	// trlix.Filter(func(et *etable.Table, row int) bool {
	// 	return et.CellFloat("ActMatch", row) > 0 // only correct trials
	// })
	gpsp := split.GroupBy(trlix, []string{"GenAction"})
	split.Agg(gpsp, "ActMatch", agg.AggMean)
	for _, lnm := range ss.PulvLays {
		_, err := split.AggTry(gpsp, lnm+"_CosDiff", agg.AggMean)
		if err != nil {
			log.Println(err)
		}
	}
	ss.TrnErrStats = gpsp.AggsToTable(etable.ColNameOnly)

	agsp := split.All(trlix)
	for _, lnm := range ss.TrainEnv.Inters {
		split.Agg(agsp, lnm, agg.AggMean)
	}
	ss.TrnAggStats = agsp.AggsToTable(etable.ColNameOnly)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("ActMatch", row, ss.EpcActMatch)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

	for _, lnm := range ss.TrainEnv.Acts {
		rw := ss.TrnErrStats.RowsByString("GenAction", lnm, etable.Equals, etable.UseCase)
		if len(rw) > 0 {
			dt.SetCellFloat(lnm+"Cor", row, ss.TrnErrStats.CellFloat("ActMatch", rw[0]))
		}
	}

	for _, lnm := range ss.TrainEnv.Inters {
		dt.SetCellFloat(lnm, row, ss.TrnAggStats.CellFloat(lnm, 0))
	}

	for li, lnm := range ss.PulvLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(lnm+"_CosDiff", row, agg.Agg(trlix, lnm+"_CosDiff", agg.AggMean)[0])
		dt.SetCellFloat(lnm+"_MaxGeM", row, float64(ly.ActAvg.AvgMaxGeM))
		dt.SetCellFloat(lnm+"_ActAvg", row, float64(ly.ActAvg.ActMAvg))
		for _, act := range ss.CosDifActs {
			arow := ss.TrnErrStats.RowsByString("GenAction", act, etable.Equals, etable.UseCase)
			if len(arow) != 1 {
				continue
			}
			val := ss.TrnErrStats.Cols[2+li].FloatVal1D(arow[0])
			dt.SetCellFloat(lnm+"_CosDiff_"+act, row, val)
		}
	}

	for _, lnm := range ss.HidLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		hog, dead := ss.HogDead(lnm)
		dt.SetCellFloat(lnm+"_Dead", row, dead)
		dt.SetCellFloat(lnm+"_Hog", row, hog)
		dt.SetCellFloat(lnm+"_MaxGeM", row, float64(ly.ActAvg.AvgMaxGeM))
		dt.SetCellFloat(lnm+"_ActAvg", row, float64(ly.ActAvg.ActMAvg))
	}

	for _, lnm := range ss.InputLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(lnm+"_ActAvg", row, float64(ly.ActAvg.ActMAvg))
	}

	if ss.RepsInterval > 0 && epc%ss.RepsInterval == 0 {
		reps := etable.NewIdxView(ss.TrnTrlRepLog)
		if ss.UseMPI {
			empi.GatherTableRows(ss.TrnTrlRepLogAll, ss.TrnTrlRepLog, ss.Comm)
			reps = etable.NewIdxView(ss.TrnTrlRepLogAll)
		}
		// reps.SortColName("Obj", true)
		for _, lnm := range ss.HidLays {
			ss.PCA.TableCol(reps, lnm, metric.Covariance64)
			var nstr float64
			ln := len(ss.PCA.Values)
			for i, v := range ss.PCA.Values {
				// fmt.Printf("%s\t\t %d  %g\n", lnm, i, v)
				if v >= 0.01 {
					nstr = float64(ln - i)
					break
				}
			}
			var top5, next5 float64
			for i := 0; i < 5; i++ {
				top5 += ss.PCA.Values[ln-1-i]
				next5 += ss.PCA.Values[ln-6-i]
			}
			sum := norm.Sum64(ss.PCA.Values)
			ravg := (sum - (top5 + next5)) / float64(ln-10)
			dt.SetCellFloat(lnm+"_PCA_NStrong", row, nstr)
			dt.SetCellFloat(lnm+"_PCA_Top5", row, top5/5)
			dt.SetCellFloat(lnm+"_PCA_Next5", row, next5/5)
			dt.SetCellFloat(lnm+"_PCA_Rest", row, ravg)
		}
	} else {
		if row > 0 {
			for _, lnm := range ss.HidLays {
				dt.SetCellFloat(lnm+"_PCA_NStrong", row, dt.CellFloat(lnm+"_PCA_NStrong", row-1))
				dt.SetCellFloat(lnm+"_PCA_Top5", row, dt.CellFloat(lnm+"_PCA_Top5", row-1))
				dt.SetCellFloat(lnm+"_PCA_Next5", row, dt.CellFloat(lnm+"_PCA_Next5", row-1))
				dt.SetCellFloat(lnm+"_PCA_Rest", row, dt.CellFloat(lnm+"_PCA_Rest", row-1))
			}
		}
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}

	trl.SetNumRows(0)
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"ActMatch", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TrainEnv.Acts {
		sch = append(sch, etable.Column{lnm + "Cor", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.TrainEnv.Inters {
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.PulvLays {
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_MaxGeM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
		for _, act := range ss.CosDifActs {
			sch = append(sch, etable.Column{lnm + "_CosDiff_" + act, etensor.FLOAT64, nil, nil})
		}
	}
	for _, lnm := range ss.HidLays {
		sch = append(sch, etable.Column{lnm + "_Dead", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Hog", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_MaxGeM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_NStrong", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_Top5", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_Next5", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PCA_Rest", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.InputLays {
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("ActMatch", eplot.On, eplot.FixMin, 0, eplot.FixMax, .25)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.TrainEnv.Acts {
		plt.SetColParams(lnm+"Cor", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
	for _, lnm := range ss.TrainEnv.Inters {
		plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
	for _, lnm := range ss.PulvLays {
		plt.SetColParams(lnm+"_CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"_MaxGeM", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .25)
		for _, act := range ss.CosDifActs {
			plt.SetColParams(lnm+"_CosDiff_"+act, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		}
	}
	for _, lnm := range ss.HidLays {
		plt.SetColParams(lnm+"_Dead", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"_Hog", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"_MaxGeM", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .25)
	}
	for _, lnm := range ss.InputLays {
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .25)
	}

	return plt
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	env := &ss.TrainEnv

	dt.SetCellFloat("Run", row, float64(env.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(env.Epoch.Cur))
	dt.SetCellFloat("Event", row, float64(env.Event.Cur))
	dt.SetCellFloat("X", row, float64(env.PosI.X))
	dt.SetCellFloat("Y", row, float64(env.PosI.Y))
	dt.SetCellString("NetAction", row, ss.NetAction)
	dt.SetCellString("GenAction", row, ss.GenAction)
	dt.SetCellString("ActAction", row, ss.ActAction)
	dt.SetCellFloat("ActMatch", row, ss.ActMatch)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for i, lnm := range ss.PulvLays {
		dt.SetCellFloat(lnm+"_CosDiff", row, float64(ss.TrlCosDiffTRC[i]))
	}
	for _, lnm := range ss.TrainEnv.Inters {
		dt.SetCellFloat(lnm, row, float64(ss.TrainEnv.InterStates[lnm]))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of trials while training, including position")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Event", etensor.INT64, nil, nil},
		{"X", etensor.FLOAT64, nil, nil},
		{"Y", etensor.FLOAT64, nil, nil},
		{"NetAction", etensor.STRING, nil, nil},
		{"GenAction", etensor.STRING, nil, nil},
		{"ActAction", etensor.STRING, nil, nil},
		{"ActMatch", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.PulvLays {
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.TrainEnv.Inters {
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Event Plot"
	plt.Params.XAxisCol = "Event"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Event", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("NetAction", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GenAction", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("ActAction", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("ActMatch", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.PulvLays {
		plt.SetColParams(lnm+"_CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	for _, lnm := range ss.TrainEnv.Inters {
		plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
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
	dt.SetCellString("NetAction", row, ss.NetAction)
	dt.SetCellString("GenAction", row, ss.GenAction)
	dt.SetCellString("ActAction", row, ss.ActAction)
	dt.SetCellFloat("ActMatch", row, ss.ActMatch)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.TrainEnv.Inters {
		dt.SetCellFloat(lnm, row, float64(ss.TrainEnv.InterStates[lnm]))
	}
	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Event", etensor.INT64, nil, nil},
		{"X", etensor.FLOAT64, nil, nil},
		{"Y", etensor.FLOAT64, nil, nil},
		{"NetAction", etensor.STRING, nil, nil},
		{"GenAction", etensor.STRING, nil, nil},
		{"ActAction", etensor.STRING, nil, nil},
		{"ActMatch", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TrainEnv.Inters {
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Test Trial Plot"
	plt.Params.XAxisCol = "Event"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, true, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, true, 0, eplot.FloatMax, 0)
	plt.SetColParams("Event", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("NetAction", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GenAction", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("ActAction", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("ActMatch", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.TrainEnv.Inters {
		plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TstTrlLog
	trlix := etable.NewIdxView(trl)

	gpsp := split.GroupBy(trlix, []string{"GenAction"})
	split.Agg(gpsp, "ActMatch", agg.AggMean)
	ss.TrnErrStats = gpsp.AggsToTable(etable.ColNameOnly)

	agsp := split.All(trlix)
	for _, lnm := range ss.TrainEnv.Inters {
		split.Agg(agsp, lnm, agg.AggMean)
	}
	ss.TrnAggStats = agsp.AggsToTable(etable.ColNameOnly)

	trl.SetNumRows(0)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	for _, lnm := range ss.TrainEnv.Acts {
		rw := ss.TrnErrStats.RowsByString("GenAction", lnm, etable.Equals, etable.UseCase)
		if len(rw) > 0 {
			dt.SetCellFloat(lnm+"Cor", row, ss.TrnErrStats.CellFloat("ActMatch", rw[0]))
		}
	}

	for _, lnm := range ss.TrainEnv.Inters {
		dt.SetCellFloat(lnm, row, ss.TrnAggStats.CellFloat(lnm, 0))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
	if ss.TstEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == ss.MaxEpcs {
			dt.WriteCSVHeaders(ss.TstEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TstEpcFile, row, etable.Tab)
	}
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

	for _, lnm := range ss.TrainEnv.Acts {
		sch = append(sch, etable.Column{lnm + "Cor", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.TrainEnv.Inters {
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)

	for _, lnm := range ss.TrainEnv.Acts {
		plt.SetColParams(lnm+"Cor", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
	for _, lnm := range ss.TrainEnv.Inters {
		plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " Ge.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Act.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", true, true, 0, true, .5)
		plt.SetColParams(lnm+" Act.Avg", true, true, 0, true, .5)
	}
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

	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	// runix := etable.NewIdxView(dt)
	// spl := split.GroupBy(runix, []string{"Params"})
	// split.Desc(spl, "PctCor")
	// ss.RunStats = spl.AggsToTable(false)

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

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.Scene().Camera.Pose.Pos.Set(0, 2.2, 1.9)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigWorldGui configures all the world view GUI elements
func (ss *Sim) ConfigWorldGui() *gi.Window {
	// order: Empty, wall, food, water, foodwas, waterwas
	ss.MatColors = []string{"lightgrey", "black", "orange", "blue", "brown", "navy"}

	ss.Trace = ss.TrainEnv.World.Clone().(*etensor.Int)

	width := 1600
	height := 1200

	win := gi.NewMainWindow("fworld", "Flat World", width, height)
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

	tbar.AddAction(gi.ActOpts{Label: "Backward", Icon: "wedge-down", Tooltip: "Step Backward", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Backward()
		vp.SetFullReRender()
	})

	tbar.AddSeparator("sep-eat")

	tbar.AddAction(gi.ActOpts{Label: "Eat", Icon: "field", Tooltip: "Eat food -- only if directly in front", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Eat()
		vp.SetFullReRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Drink", Icon: "svg", Tooltip: "Drink water -- only if directly in front", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Drink()
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
	cnm := "FWorldColors"
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
	}

	nc := len(ss.TrainEnv.Mats)
	ss.Trace.Set([]int{ss.TrainEnv.PosI.Y, ss.TrainEnv.PosI.X}, nc+ss.TrainEnv.Angle/ss.TrainEnv.AngInc)

	updt := ss.WorldTabs.UpdateStart()
	ss.TraceView.UpdateSig()
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

func (ss *Sim) Backward() {
	ss.TrainEnv.Action("Backward", nil)
	ss.UpdateWorldGui()
}

func (ss *Sim) Eat() {
	ss.TrainEnv.Action("Eat", nil)
	ss.UpdateWorldGui()
}

func (ss *Sim) Drink() {
	ss.TrainEnv.Action("Drink", nil)
	ss.UpdateWorldGui()
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("Emery")
	gi.SetAppAbout(`Full brain predictive learning in navigational / survival environment. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("Emery", "Emery simulated rat / cat", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	stb := tv.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	ss.ConfigSpikeRasts()
	for _, lnm := range ss.SpikeRecLays {
		sr := ss.SpikeRastTsr(lnm)
		tg := ss.SpikeRastGrid(lnm)
		tg.SetName(lnm + "Spikes")
		gi.AddNewLabel(stb, lnm, lnm+":")
		stb.AddChild(tg)
		gi.AddNewSpace(stb, lnm+"_spc")
		ss.ConfigSpikeGrid(tg, sr)
	}

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.3, .7)

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

	tbar.AddSeparator("test")

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

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on change -- wrap
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

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddAction(gi.ActOpts{Label: "Reset TrlLog", Icon: "reset", Tooltip: "Reset the accumulated trial log"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TrnTrlLog.SetNumRows(0)
			ss.TrnTrlPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/deep_fsa/README.md")
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

	/*
		inQuitPrompt := false
		gi.SetQuitReqFunc(func() {
			if inQuitPrompt {
				return
			}
			inQuitPrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
				Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, true, true,
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
				Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, true, true,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inClosePrompt = false
					}
				})
		})
	*/

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
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&ss.SaveARFs, "arfs", false, "if true, save final arfs after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.BoolVar(&ss.UseMPI, "mpi", false, "if set, use MPI for distributed computation")
	flag.Parse()
	ss.Init()

	if ss.UseMPI {
		ss.MPIInit()
	}

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
			ss.TrnEpcFile = nil
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

// MPIInit initializes MPI
func (ss *Sim) MPIInit() {
	mpi.Init()
	var err error
	ss.Comm, err = mpi.NewComm(nil) // use all procs
	if err != nil {
		log.Println(err)
		ss.UseMPI = false
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.UseMPI {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
func (ss *Sim) CollectDWts(net *axon.Network) {
	net.CollectDWts(&ss.AllDWts)
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	if ss.UseMPI {
		ss.CollectDWts(&ss.Net.Network)
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ss.SumDWts, mpi.WorldSize())
	}
	ss.Net.WtFmDWt()
}
