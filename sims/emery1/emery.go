// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// emery1 is a simulated virtual rodent
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/emer/emergent/actrf"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor" // include to get gui views
	"github.com/emer/etable/etview"
	"github.com/emer/etable/split"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/gist"
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
			{Sel: "Layer", Desc: "using default 1.8 inhib for hidden layers",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":  "1.8",
					"Layer.Learn.AvgL.Gain": "2.5",
					"Layer.Act.Gbar.L":      "0.2",
				}},
			{Sel: ".BurstTRC", Desc: "standard weight is .3 here for larger distributed reps. no learn",
				Params: params.Params{
					"Prjn.WtInit.Mean": "0.3",
					"Prjn.WtInit.Var":  "0",
					"Prjn.Learn.Learn": "false",
				}},
			{Sel: ".BurstCtxt", Desc: "no weight balance on deep context prjns -- makes a diff!",
				Params: params.Params{
					"Prjn.Learn.WtBal.On": "false",
				}},
			{Sel: ".V2d", Desc: "depth input layers use pool inhibition, weaker global?",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.2", // some weaker global inhib
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.8",
				}},
			{Sel: ".S1", Desc: "S1 uses pool inhib",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.0", // some weaker global inhib
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.4", // weaker
				}},
			{Sel: ".MT", Desc: "MT uses pool inhibition, full global?",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.6", // 1.4 too low,
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.6",
				}},
			{Sel: ".7a", Desc: "IPL uses pool inhibition, weaker global?",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.2",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.6",
				}},
			{Sel: ".PCC", Desc: "PCC uses pool inhibition but is treated as full",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.4",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.6",
				}},
			{Sel: "#PCCP", Desc: "no topo",
				Params: params.Params{
					"Layer.TRC.NoTopo": "true", // true def
				}},
			{Sel: "#SMAP", Desc: "no topo",
				Params: params.Params{
					"Layer.TRC.NoTopo": "true", // true def
				}},
			{Sel: "#M1P", Desc: "M1P uses pool inhib",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.0", // some weaker global inhib
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.6", // weaker
				}},
			{Sel: "#M1", Desc: "noise!",
				Params: params.Params{
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.01",
					"Layer.Act.Noise.Type": "GeNoise",
					"Layer.Inhib.Pool.Gi":  "1.8", // reg
				}},
			{Sel: "#SMA", Desc: "noise!",
				Params: params.Params{
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.01",
					"Layer.Act.Noise.Type": "GeNoise",
				}},

			//////////////////////////////////////////////////////////
			// Prjns

			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.1",
				}},
			{Sel: ".CTBack", Desc: "deep top-down -- stronger",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.5",
				}},
			{Sel: ".Lateral", Desc: "default for lateral",
				Params: params.Params{
					"Prjn.WtInit.Sym":  "false",
					"Prjn.WtScale.Rel": "0.02", // .02 > .05 == .01 > .1  -- very minor diffs on TE cat
					"Prjn.WtInit.Mean": "0.5",
					"Prjn.WtInit.Var":  "0",
				}},
			{Sel: ".CTFmSuper", Desc: "CT from main super -- fixed one2one",
				Params: params.Params{
					"Prjn.WtInit.Mean": "0.8", // 0.8 better for wwi3d, 0.5 default
					"Prjn.WtScale.Rel": "1",
				}},
			{Sel: ".CTSelf", Desc: "CT to CT",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.5",
				}},
			{Sel: ".FwdToPulv", Desc: "feedforward to pulvinar directly",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.1",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "500",
				}},
		},
	}},
	{Name: "NoMomentum", Desc: "no momentum or normalization", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no norm or momentum",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.Momentum.On": "false",
				}},
		},
	}},
	{Name: "WtBalOn", Desc: "try with weight bal on", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "weight bal on",
				Params: params.Params{
					"Prjn.Learn.WtBal.On": "true",
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
	Net              *deep.Network     `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	CurPosMap        *etensor.Float32  `view:"current pos map from env"`
	PosAFs           actrf.RFs         `view:"no-inline" desc:"activation-based receptive fields for position target"`
	TrnEpcLog        *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TrnTrlLog        *etable.Table     `view:"no-inline" desc:"training trial-level log data"`
	TstEpcLog        *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog        *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog        *etable.Table     `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats      *etable.Table     `view:"no-inline" desc:"stats on test trials where errors were made"`
	TstCycLog        *etable.Table     `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog           *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats         *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	Params           params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet         string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag              string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Prjn4x4Skp2      *prjn.PoolTile    `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x4Skp2Recip *prjn.PoolTile    `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn3x3Skp1      *prjn.PoolTile    `view:"no-inline" desc:"feedforward 3x3 skip 1 topo prjn"`
	Prjn4x4Skp4      *prjn.PoolTile    `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn"`
	Prjn4x4Skp4Recip *prjn.PoolTile    `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn, recip"`
	MaxRuns          int               `desc:"maximum number of model runs to perform"`
	MaxEpcs          int               `desc:"maximum number of epochs to run per model run"`
	NZeroStop        int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv         FWorld            `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time             leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn           bool              `desc:"whether to update the network view while running"`
	TrainUpdt        leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt         leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval     int               `desc:"how often to run through all the test patterns, in terms of training epochs"`
	LayStatNms       []string          `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	PosAFNms         []string          `desc:"names of layers to compute position activation fields on"`

	// statistics: note use float64 as that is best for etable.Table
	PulvLays      []string  `view:"-" desc:"pulvinar layers -- for stats"`
	HidLays       []string  `view:"-" desc:"hidden layers: super and CT -- for hogging stats"`
	SuperLays     []string  `view:"-" desc:"superficial layers"`
	TrlSSE        float64   `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE     float64   `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff    float64   `inactive:"+" desc:"current trial's overall cosine difference"`
	TrlCosDiffTRC []float64 `inactive:"+" desc:"current trial's cosine difference for pulvinar (TRC) layers"`
	EpcSSE        float64   `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE     float64   `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr     float64   `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcPctCor     float64   `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcCosDiff    float64   `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcCosDiffTRC []float64 `inactive:"+" desc:"last epoch's average cosine difference for TRC layers (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	FirstZero     int       `inactive:"+" desc:"epoch at when SSE first went to zero"`
	NZero         int       `inactive:"+" desc:"number of epochs in a row with zero SSE"`

	// internal state - view:"-"
	SumSSE        float64            `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE     float64            `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff    float64            `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiffTRC []float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch, per TRC"`
	CntErr        int                `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	Win           *gi.Window         `view:"-" desc:"main GUI window"`
	NetView       *netview.NetView   `view:"-" desc:"the network viewer"`
	ToolBar       *gi.ToolBar        `view:"-" desc:"the master toolbar"`
	WorldWin      *gi.Window         `view:"-" desc:"FWorld GUI window"`
	WorldTabs     *gi.TabView        `view:"-" desc:"FWorld TabView"`
	MatColors     []string           `desc:"color strings in material order"`
	Trace         *etensor.Int       `view:"no-inline" desc:"trace of movement for visualization"`
	TraceView     *etview.TensorGrid `desc:"view of the activity trace"`
	WorldView     *etview.TensorGrid `desc:"view of the world"`
	State         *etable.Table      `desc:"table recording env"`
	StateView     *etview.TableView  `view:"-" desc:"the main view"`
	TrnEpcPlot    *eplot.Plot2D      `view:"-" desc:"the training epoch plot"`
	TrnTrlPlot    *eplot.Plot2D      `view:"-" desc:"the training trial plot"`
	TstEpcPlot    *eplot.Plot2D      `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot    *eplot.Plot2D      `view:"-" desc:"the test-trial plot"`
	TstCycPlot    *eplot.Plot2D      `view:"-" desc:"the test-cycle plot"`
	RunPlot       *eplot.Plot2D      `view:"-" desc:"the run plot"`
	TrnEpcFile    *os.File           `view:"-" desc:"log file"`
	RunFile       *os.File           `view:"-" desc:"log file"`
	PosAFTsr      etensor.Float32    `view:"-" desc:"for holding layer values"`
	ActVals       etensor.Float32    `view:"-" desc:"for action vals"`
	PopVals       []float32          `view:"-" desc:"tmp pop code values"`
	SaveWts       bool               `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui         bool               `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams  bool               `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning     bool               `view:"-" desc:"true if sim is running"`
	StopNow       bool               `view:"-" desc:"flag to stop running"`
	NeedsNewRun   bool               `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed       int64              `view:"-" desc:"the current random seed"`
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
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.Quarter // leabra.AlphaCycle
	ss.TestUpdt = leabra.Cycle
	ss.TestInterval = 500
	ss.LayStatNms = []string{"MT", "MTCT", "SMA", "SMACT"}
	ss.PosAFNms = []string{"PCC", "PCCCT"}
	ss.NewPrjns()
}

// NewPrjns creates new projections
func (ss *Sim) NewPrjns() {
	ss.Prjn4x4Skp2 = prjn.NewPoolTile()
	ss.Prjn4x4Skp2.Size.Set(4, 4)
	ss.Prjn4x4Skp2.Skip.Set(2, 2)
	ss.Prjn4x4Skp2.Start.Set(-1, -1)
	ss.Prjn4x4Skp2Recip = prjn.NewPoolTileRecip(ss.Prjn4x4Skp2)
	ss.Prjn3x3Skp1 = prjn.NewPoolTile()
	ss.Prjn3x3Skp1.Size.Set(3, 3)
	ss.Prjn3x3Skp1.Skip.Set(1, 1)
	ss.Prjn3x3Skp1.Start.Set(-1, -1)
	ss.Prjn4x4Skp4 = prjn.NewPoolTile()
	ss.Prjn4x4Skp4.Size.Set(4, 4)
	ss.Prjn4x4Skp4.Skip.Set(4, 4)
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
		ss.MaxEpcs = 500
		ss.NZeroStop = -1
	}

	ss.TrainEnv.Config(500) // n trials per epoch
	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Run.Max = ss.MaxRuns
	ss.TrainEnv.Init(0)
	ss.TrainEnv.Validate()
}

func (ss *Sim) ConfigNet(net *deep.Network) {
	net.InitName(net, "Emery")

	// input / output layers:
	v2 := net.AddLayer4D("V2d", 8, 16, 1, ss.TrainEnv.NFOVRays, emer.Input)
	s1 := net.AddLayer4D("S1", 4, 1, 1, ss.TrainEnv.PopSize, emer.Input)

	m1 := net.AddLayer2D("M1", 7, 7, emer.Hidden)
	m1p := net.AddLayer2D("M1P", ss.TrainEnv.PatSize.Y, ss.TrainEnv.PatSize.X, emer.Target)

	mt, mtct, mtp := net.AddDeep4D("MT", 4, 8, 4, 4)
	mtp.Shape().SetShape([]int{4, 8, 1, ss.TrainEnv.NFOVRays}, nil, nil)
	mtp.(*deep.TRCLayer).Drivers.Add("V2d")

	ipl, iplct, iplp := net.AddDeep4D("7a", 2, 4, 5, 5)
	iplp.Shape().SetShape([]int{2, 4, 3, ss.TrainEnv.NFOVRays}, nil, nil)
	iplp.(*deep.TRCLayer).Drivers.Add("V2d", "S1") // V2d = NDepthCode = 8, S1 = AngRes = 16,

	pcc, pccct, pccp := net.AddDeep4D("PCC", 2, 4, 5, 5)
	pccp.Shape().SetShape([]int{2, 4, 3, ss.TrainEnv.NFOVRays}, nil, nil)
	pccp.(*deep.TRCLayer).Drivers.Add("V2d", "S1") // V2d = NDepthCode = 8, S1 = AngRes = 16,
	// note: has Layer.TRC.NoTopo set to true in params by default

	sma, smact, smap := net.AddDeep2D("SMA", 8, 8)
	smap.Shape().SetShape([]int{m1.Shape().Dim(0) + 1, ss.TrainEnv.PopSize}, nil, nil)
	smap.(*deep.TRCLayer).Drivers.Add("M1", "S1")

	m1.SetClass("M1")
	m1p.SetClass("M1")

	mt.SetClass("MT")
	mtct.SetClass("MT")
	mtp.SetClass("MT")

	ipl.SetClass("7a")
	iplct.SetClass("7a")
	iplp.SetClass("7a")

	pcc.SetClass("PCC")
	pccct.SetClass("PCC")
	pccp.SetClass("PCC")

	sma.SetClass("SMA")
	smact.SetClass("SMA")
	smap.SetClass("SMA")

	s1.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "V2d", XAlign: relpos.Left, Space: 4})

	mt.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "V2d", XAlign: relpos.Left, YAlign: relpos.Front})
	mtct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "MT", XAlign: relpos.Left, Space: 2})
	mtp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "MTCT", XAlign: relpos.Left, Space: 2})

	ipl.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "MT", YAlign: relpos.Front})
	iplct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "7a", XAlign: relpos.Left, Space: 2})
	iplp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "7aCT", XAlign: relpos.Left, Space: 2})

	pcc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "7a", YAlign: relpos.Front, Space: 10})
	pccct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "PCC", XAlign: relpos.Left, Space: 2})
	pccp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "PCCCT", XAlign: relpos.Left, Space: 2})

	sma.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "PCC", YAlign: relpos.Front, Space: 10})
	smact.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "SMA", XAlign: relpos.Left, Space: 2})
	smap.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "SMACT", XAlign: relpos.Left, Space: 2})

	m1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "SMA", YAlign: relpos.Front, Space: 8})
	m1p.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "M1", XAlign: relpos.Left, Space: 8})

	full := prjn.NewFull()
	sameu := prjn.NewPoolSameUnit()
	sameu.SelfCon = false

	////////////////////
	// basic super cons

	net.ConnectLayers(v2, mt, ss.Prjn4x4Skp2, emer.Forward)

	// MT <-> IPL
	net.ConnectLayers(mt, ipl, ss.Prjn4x4Skp2, emer.Forward)
	net.ConnectLayers(ipl, mt, ss.Prjn4x4Skp2Recip, emer.Back)

	// IPL <-> PCC
	net.ConnectLayers(ipl, pcc, full, emer.Forward)
	net.ConnectLayers(pcc, ipl, full, emer.Back)

	// PCC <-> SMA
	net.ConnectLayers(pcc, sma, full, emer.Forward)
	net.ConnectLayers(sma, pcc, full, emer.Back)

	// SMA <-> M1
	net.ConnectLayers(sma, m1, full, emer.Forward)

	net.BidirConnectLayers(m1, m1p, full)

	////////////////////
	// to MT

	net.ConnectLayers(mt, mt, sameu, emer.Lateral)

	net.ConnectLayers(sma, mt, full, emer.Back)
	net.ConnectLayers(s1, mt, full, emer.Back)

	net.ConnectCtxtToCT(mtct, mtct, ss.Prjn3x3Skp1).SetClass("CTSelf")

	// MTCT top-down depth
	net.ConnectLayers(iplct, mtct, ss.Prjn4x4Skp2Recip, emer.Back).SetClass("CTBack")
	net.ConnectLayers(pccct, mtct, full, emer.Back).SetClass("CTBack")

	// todo: try S -> CT leak back -- useful in wwi3d
	// todo: try higher CT -> mtp  -- useful in wwi3d

	////////////////////
	// to IPL / 7a

	net.ConnectLayers(sma, ipl, full, emer.Back)
	net.ConnectLayers(s1, ipl, full, emer.Back)
	net.ConnectLayers(m1p, ipl, full, emer.Back) // todo: m1?

	net.ConnectCtxtToCT(iplct, iplct, full).SetClass("CTSelf")

	net.ConnectLayers(pccct, iplct, full, emer.Back).SetClass("CTBack")
	// net.ConnectLayers(smact, iplct, full, emer.Back).SetClass("CTBack")

	net.ConnectLayers(mtct, iplp, ss.Prjn4x4Skp2, emer.Forward).SetClass("FwdToPulv")

	// todo: try S -> CT leak back -- useful in wwi3d
	// todo: try higher CT -> mtp  -- useful in wwi3d

	////////////////////
	// to PCC

	net.ConnectLayers(s1, pcc, full, emer.Forward)
	net.ConnectLayers(m1p, pcc, full, emer.Back)

	net.ConnectCtxtToCT(pccct, pccct, full).SetClass("CTSelf")

	net.ConnectLayers(smact, pccct, full, emer.Back).SetClass("CTBack")

	////////////////////
	// to SMA

	net.ConnectLayers(ipl, sma, full, emer.Forward) // todo: forward??
	net.ConnectLayers(m1p, sma, full, emer.Back)

	net.ConnectCtxtToCT(smact, smact, full).SetClass("CTSelf")

	net.ConnectLayers(m1p, smact, full, emer.Back)

	////////////////////
	// to M1

	net.ConnectLayers(smact, m1p, full, emer.Forward)
	net.ConnectLayers(sma, m1p, full, emer.Forward)

	ss.PulvLays = make([]string, 0, 10)
	ss.HidLays = make([]string, 0, 10)
	ss.SuperLays = make([]string, 0, 10)
	for _, ly := range net.Layers {
		if ly.IsOff() {
			continue
		}
		switch ly.Type() {
		case deep.TRC:
			ss.PulvLays = append(ss.PulvLays, ly.Name())
		case emer.Hidden:
			ss.SuperLays = append(ss.SuperLays, ly.Name())
			fallthrough
		case deep.CT:
			ss.HidLays = append(ss.HidLays, ly.Name())
		}
	}
	ss.PulvLays = append(ss.PulvLays, "M1P")

	// using 4 total threads -- todo: didn't work
	/*
		mt.SetThread(1)
		mtct.SetThread(1)
		ipl.SetThread(2)
		iplct.SetThread(2)
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
	ss.InitWts(net)
}

// Initialize network weights including scales
func (ss *Sim) InitWts(net *deep.Network) {
	net.InitTopoScales() //  sets all wt scales
	net.InitWts()
	net.LrateMult(1) // restore initial learning rate value
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
	return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tEvent:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Event.Cur, ss.Time.Cycle, ss.TrainEnv.Event.Cur)
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

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					ss.UpdateView(train)
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
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(net *deep.Network, en env.Env) {
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "Vestibular", "Inters", "Action"}
	lays := []string{"V2d", "V2d", "V1fov", "S1", "S1v", "Insula", "M1"}
	for i, lnm := range lays {
		lyi := ss.Net.LayerByName(lnm)
		if lyi == nil {
			continue
		}
		ly := lyi.(leabra.LeabraLayer).AsLeabra()
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

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		ss.TrainEnv.Event.Cur = 0
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
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

	ss.ApplyInputs(ss.Net, &ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
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
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	ss.CntErr = 0
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// TrialStatsTRC computes the trial-level statistics for TRC layers
func (ss *Sim) TrialStatsTRC(accum bool) {
	nt := len(ss.PulvLays)
	if len(ss.TrlCosDiffTRC) != nt {
		ss.TrlCosDiffTRC = make([]float64, nt)
		ss.SumCosDiffTRC = make([]float64, nt)
		ss.EpcCosDiffTRC = make([]float64, nt)
	}
	acd := 0.0
	for i, ln := range ss.PulvLays {
		ly := ss.Net.LayerByName(ln).(leabra.LeabraLayer).AsLeabra()
		cd := float64(ly.CosDiff.Cos)
		acd += cd
		ss.TrlCosDiffTRC[i] = cd
		if accum {
			ss.SumCosDiffTRC[i] += cd
		}
	}
	ss.TrlCosDiff = acd / 3
	if accum {
		ss.SumCosDiff += ss.TrlCosDiff
	}
}

// TrialStatsTRC computes the trial-level statistics for TRC layers
func (ss *Sim) EpochStatsTRC(nt float64) {
	acd := 0.0
	for i := range ss.PulvLays {
		ss.EpcCosDiffTRC[i] = ss.SumCosDiffTRC[i] / nt
		ss.SumCosDiffTRC[i] = 0
		acd += ss.EpcCosDiffTRC[i]
	}
	ss.EpcCosDiff = acd / 3
}

// UpdtPosAFs updates position activation rf's
func (ss *Sim) UpdtPosAFs() {
	naf := len(ss.PosAFNms)
	if len(ss.PosAFs.RFs) != naf {
		for _, lnm := range ss.PosAFNms {
			ly := ss.Net.LayerByName(lnm)
			if ly == nil {
				continue
			}
			ly.UnitValsTensor(&ss.PosAFTsr, "ActM")
			af := ss.PosAFs.AddRF(lnm, &ss.PosAFTsr, ss.CurPosMap)
			af.NormRF.SetMetaData("min", "0")
			af.NormRF.SetMetaData("colormap", "JetMuted")
		}
	}
	for _, lnm := range ss.PosAFNms {
		ly := ss.Net.LayerByName(lnm)
		if ly == nil {
			continue
		}
		ly.UnitValsTensor(&ss.PosAFTsr, "ActM")
		ss.PosAFs.Add(lnm, &ss.PosAFTsr, ss.CurPosMap, 0.01) // thr prevent weird artifacts
	}
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	// ss.TrlCosDiff = float64(inp.CosDiff.Cos)
	// ss.TrlSSE, ss.TrlAvgSSE = inp.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if accum {
		// ss.SumSSE += ss.TrlSSE
		// ss.SumAvgSSE += ss.TrlAvgSSE
		if ss.TrlSSE != 0 {
			ss.CntErr++
		}
	}
	ss.TrialStatsTRC(accum)
	// ss.UpdtPosAFs()
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
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	// ss.TestEnv.Step()
	//
	// // Query counters FIRST
	// _, _, chg := ss.TestEnv.Counter(env.Epoch)
	// if chg {
	// 	if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
	// 		ss.UpdateView(false)
	// 	}
	// 	ss.LogTstEpc(ss.TstEpcLog)
	// 	if returnOnChg {
	// 		return
	// 	}
	// }
	//
	// ss.ApplyInputs(&ss.TestEnv)
	// ss.AlphaCyc(false)   // !train
	// ss.TrialStats(false) // !accumulate
	// ss.LogTstTrl(ss.TstTrlLog)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	// ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on change -- don't wrap
		// _, _, chg := ss.TestEnv.Counter(env.Epoch)
		// if chg || ss.StopNow {
		// 	break
		// }
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
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  TrnEpcLog

// HogDead computes the proportion of units in given layer name with ActAvg over hog thr
// and under dead threshold
func (ss *Sim) HogDead(lnm string) (hog, dead float64) {
	ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
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
	nt := float64(ss.TrainEnv.Event.Prv)

	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.CntErr) / nt
	ss.CntErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpochStatsTRC(nt)
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}
	if ss.EpcPctErr == 0 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

	for i, lnm := range ss.PulvLays {
		dt.SetCellFloat(lnm+" CosDiff", row, float64(ss.EpcCosDiffTRC[i]))
	}

	for _, lnm := range ss.HidLays {
		hog, dead := ss.HogDead(lnm)
		dt.SetCellFloat(lnm+"_Dead", row, dead)
		dt.SetCellFloat(lnm+"_Hog", row, hog)
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
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.PulvLays {
		sch = append(sch, etable.Column{lnm + " CosDiff", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.HidLays {
		sch = append(sch, etable.Column{lnm + "_Dead", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Hog", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1) // default plot
	plt.SetColParams("PctCor", false, true, 0, true, 1) // default plot
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	for _, lnm := range ss.PulvLays {
		plt.SetColParams(lnm+" CosDiff", true, true, 0, true, 1)
	}
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", false, true, 0, true, .5)
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
	dt.SetCellString("Action", row, env.Acts[env.Act])
	// forced := ""
	// if env.CurAct != env.ExtAct {
	// 	forced = navenv.ActionsCode[env.CurAct]
	// }
	// dt.SetCellString("Forced", row, forced)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for i, lnm := range ss.PulvLays {
		dt.SetCellFloat(lnm+" CosDiff", row, float64(ss.TrlCosDiffTRC[i]))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of trials while training, including position")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TrainEnv.Event.Prv
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Event", etensor.INT64, nil, nil},
		{"X", etensor.FLOAT64, nil, nil},
		{"Y", etensor.FLOAT64, nil, nil},
		{"Action", etensor.STRING, nil, nil},
		{"ActMag", etensor.FLOAT64, nil, nil},
		{"NetAct", etensor.STRING, nil, nil},
		{"Forced", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.PulvLays {
		sch = append(sch, etable.Column{lnm + " CosDiff", etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Event Plot"
	plt.Params.XAxisCol = "X"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("Event", false, true, 0, false, 0)
	plt.SetColParams("X", false, true, 0, true, 1)
	plt.SetColParams("Y", true, true, 0, true, 1)
	plt.SetColParams("Action", false, true, 0, false, 0)
	plt.SetColParams("ActMag", false, true, 0, true, 1)
	plt.SetColParams("NetAct", false, true, 0, false, 0)
	plt.SetColParams("Forced", true, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	for _, lnm := range ss.PulvLays {
		plt.SetColParams(lnm+" CosDiff", false, true, 0, true, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TrainEnv.Event.Cur
	row := trl

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	// dt.SetCellString("TrialName", row, ss.TestEnv.String())
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}
	// inp.UnitValsTensor(&ss.InputValsTsr, "ActM")
	// dt.SetCellTensor("InActM", row, ss.InputValsTsr)
	// inp.UnitValsTensor(&ss.InputValsTsr, "ActP")
	// dt.SetCellTensor("InActP", row, ss.OutputValsTsr)
	// trg.UnitValsTensor(&ss.OutputValsTsr, "ActP")
	// dt.SetCellTensor("Targs", row, ss.OutputValsTsr)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TrainEnv.Event.Prv
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	// sch = append(sch, etable.Schema{
	// 	{"InActM", etensor.FLOAT64, inp.Shp.Shp, nil},
	// 	{"InActP", etensor.FLOAT64, inp.Shp.Shp, nil},
	// 	{"Targs", etensor.FLOAT64, trg.Shp.Shp, nil},
	// }...)
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", true, true, 0, false, 0)
	plt.SetColParams("CosDiff", true, true, 0, true, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", false, true, 0, true, .5)
	}

	// plt.SetColParams("InActP", false, true, 0, true, 1)
	// plt.SetColParams("InActM", false, true, 0, true, 1)
	// plt.SetColParams("Targs", false, true, 0, true, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val > 0
	})[0])
	dt.SetCellFloat("PctCor", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val == 0
	})[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

	trlix := etable.NewIdxView(trl)
	trlix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("SSE", row) > 0 // include error trials
	})
	ss.TstErrLog = trlix.NewTable()

	allsp := split.All(trlix)
	split.Agg(allsp, "SSE", agg.AggSum)
	split.Agg(allsp, "AvgSSE", agg.AggMean)
	// split.Agg(allsp, "InActM", agg.AggMean)
	// split.Agg(allsp, "InActP", agg.AggMean)
	// split.Agg(allsp, "Targs", agg.AggMean)

	ss.TstErrStats = allsp.AggsToTable(false)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", true, true, 0, true, 1) // default plot
	plt.SetColParams("PctCor", true, true, 0, true, 1) // default plot
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
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
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
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
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(false)

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
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Emery Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("FirstZero", true, true, 0, false, 0) // default plot
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	// nv.Scene().Camera.Pose.Pos.Set(0, 1.5, 3.0) // more "head on" than default which is more "top down"
	// nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigWorldGui configures all the world view GUI elements
func (ss *Sim) ConfigWorldGui() *gi.Window {
	// order: Empty, wall, food, water, foodwas, waterwas
	ss.MatColors = []string{"lightgrey", "black", "orange", "blue", "brown", "navy"}

	ss.Trace = ss.TrainEnv.World.Clone().(*etensor.Int)

	sch := etable.Schema{
		{"TrialName", etensor.STRING, nil, nil},
		{"Depth", etensor.FLOAT32, ss.TrainEnv.CurStates["Depth"].Shape.Shp, nil},
		{"FovDepth", etensor.FLOAT32, ss.TrainEnv.CurStates["FovDepth"].Shape.Shp, nil},
		{"Fovea", etensor.FLOAT32, ss.TrainEnv.CurStates["Fovea"].Shape.Shp, nil},
		{"ProxSoma", etensor.FLOAT32, ss.TrainEnv.CurStates["ProxSoma"].Shape.Shp, nil},
		{"Vestibular", etensor.FLOAT32, ss.TrainEnv.CurStates["Vestibular"].Shape.Shp, nil},
		{"Inters", etensor.FLOAT32, ss.TrainEnv.CurStates["Inters"].Shape.Shp, nil},
		{"Action", etensor.FLOAT32, ss.TrainEnv.CurStates["Action"].Shape.Shp, nil},
	}
	ss.State = etable.NewTable("input")
	ss.State.SetFromSchema(sch, 1)
	ss.State.SetMetaData("TrialName:width", "50")

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

	sps := tv.AddNewTab(gi.KiT_SplitView, "State").(*gi.SplitView)
	sps.Dim = mat32.Y
	sps.SetStretchMax()

	ss.StateView = etview.AddNewTableView(sps, "State")
	ss.StateView.SetTable(ss.State, nil)

	ss.TraceView = etview.AddNewTensorGrid(sps, "Trace", ss.Trace)
	ss.ConfigWorldView(ss.TraceView)

	sps.SetSplits(.3, .7)

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
	if ss.WorldWin == nil {
		return
	}
	for i := 1; i < ss.State.NumCols(); i++ {
		cnm := ss.State.ColNames[i]
		inp := ss.TrainEnv.State(cnm)
		ss.State.SetCellTensor(cnm, 0, inp)
	}
	ss.State.SetCellString("TrialName", 0, ss.TrainEnv.String())

	if ss.TrainEnv.Scene.Chg { // something important happened, refresh
		ss.Trace.CopyFrom(ss.TrainEnv.World)
	}

	nc := len(ss.TrainEnv.Mats)
	ss.Trace.Set([]int{ss.TrainEnv.PosI.Y, ss.TrainEnv.PosI.X}, nc+ss.TrainEnv.Angle/ss.TrainEnv.AngInc)

	updt := ss.WorldTabs.UpdateStart()
	// ss.TraceView.UpdateSig()
	// ss.TrainEnvView.UpdateSig()
	ss.StateView.UpdateTable()
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

	win := gi.NewMainWindow("Emery", "Emery simulated rodent / cat", width, height)
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

	tbar.AddAction(gi.ActOpts{Label: "Reset PosAFs", Icon: "reset", Tooltip: "reset current position activation rfs accumulation data", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.PosAFs.Reset()
	})

	tbar.AddAction(gi.ActOpts{Label: "View PosAFs", Icon: "file-image", Tooltip: "compute current position activation rfs and view them.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.PosAFs.Avg()
		ss.PosAFs.Norm()
		for _, paf := range ss.PosAFs.RFs {
			etview.TensorGridDialog(vp, &paf.NormRF, giv.DlgOpts{Title: "Position Act RF", Prompt: paf.Name, TmpSave: nil}, nil, nil)
		}
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
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/deep_fsa/README.md")
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
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
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

func mainrun() {
	TheSim.New()
	TheSim.Config()

	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		// gi.Update2DTrace = true
		TheSim.Init()
		win := TheSim.ConfigGui()
		fwin := TheSim.ConfigWorldGui()
		fwin.GoStartEventLoop()
		win.StartEventLoop()
	}
}
