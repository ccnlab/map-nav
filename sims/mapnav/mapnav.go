// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// mapnav runs map navigation model
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/ccnlab/map-nav/sims/navenv"
	"github.com/emer/emergent/actrf"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/popcode"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor" // include to get gui views
	"github.com/emer/etable/etview"
	"github.com/emer/etable/norm"
	"github.com/emer/etable/split"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
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
			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for hidden layers",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":  "1.8",
					"Layer.Learn.AvgL.Gain": "2.5",
					"Layer.Act.Gbar.L":      "0.2",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
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
			{Sel: ".V2d", Desc: "depth input layers use pool inhibition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.2", // some weaker global inhib
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.8",
				}},
			{Sel: ".MT", Desc: "depth input layers use pool inhibition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.8",
				}},
			{Sel: ".7a", Desc: "depth input layers use pool inhibition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.8",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "500",
				}},
		},
	}},
	{Name: "DefaultInhib", Desc: "output uses default inhib instead of lower", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#Output", Desc: "go back to default",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "takes longer -- generally doesn't finish..",
				Params: params.Params{
					"Sim.MaxEpcs": "100",
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
	Net            *deep.Network     `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	CurPosMap      *etensor.Float32  `view:"current pos map from env"`
	PosAFs         actrf.RFs         `view:"no-inline" desc:"activation-based receptive fields for position target"`
	TrnEpcLog      *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog      *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog      *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog      *etable.Table     `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats    *etable.Table     `view:"no-inline" desc:"stats on test trials where errors were made"`
	TstCycLog      *etable.Table     `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog         *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats       *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	Params         params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet       string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag            string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Topo4Prjn      *prjn.PoolTile    `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Topo4PrjnRecip *prjn.PoolTile    `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Topo3Prjn      *prjn.PoolTile    `view:"no-inline" desc:"feedforward 3x3 skip 1 topo prjn"`
	Topo4S4Prjn    *prjn.PoolTile    `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn"`
	MaxRuns        int               `desc:"maximum number of model runs to perform"`
	MaxEpcs        int               `desc:"maximum number of epochs to run per model run"`
	NZeroStop      int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv       navenv.Env        `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	NDepthCode     int               `desc:"Number of units to use for population-coding depth values"`
	PopCode        popcode.OneD      `desc:"Parameters for depth population code"`
	Time           leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn         bool              `desc:"whether to update the network view while running"`
	TrainUpdt      leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt       leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval   int               `desc:"how often to run through all the test patterns, in terms of training epochs"`
	LayStatNms     []string          `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	PosAFNms       []string          `desc:"names of layers to compute position activation fields on"`

	// statistics: note use float64 as that is best for etable.Table
	TrlSSE        float64   `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE     float64   `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff    float64   `inactive:"+" desc:"current trial's overall cosine difference"`
	TRCLays       []string  `inactive:"+" desc:"TRC layer names"`
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
	SumSSE        float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE     float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff    float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiffTRC []float64        `view:"-" inactive:"+" desc:"sum to increment as we go through epoch, per TRC"`
	CntErr        int              `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	Win           *gi.Window       `view:"-" desc:"main GUI window"`
	NetView       *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar       *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	TrnEpcPlot    *eplot.Plot2D    `view:"-" desc:"the training epoch plot"`
	TstEpcPlot    *eplot.Plot2D    `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot    *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	TstCycPlot    *eplot.Plot2D    `view:"-" desc:"the test-cycle plot"`
	RunPlot       *eplot.Plot2D    `view:"-" desc:"the run plot"`
	TrnEpcFile    *os.File         `view:"-" desc:"log file"`
	RunFile       *os.File         `view:"-" desc:"log file"`
	InputValsTsr  *etensor.Float32 `view:"-" desc:"for holding layer values"`
	OutputValsTsr *etensor.Float32 `view:"-" desc:"for holding layer values"`
	PosAFTsr      *etensor.Float32 `view:"-" desc:"for holding layer values"`
	ActVals       []float32        `view:"-" desc:"for action vals"`
	PopVals       []float32        `view:"-" desc:"tmp pop code values"`
	SaveWts       bool             `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui         bool             `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams  bool             `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning     bool             `view:"-" desc:"true if sim is running"`
	StopNow       bool             `view:"-" desc:"flag to stop running"`
	NeedsNewRun   bool             `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed       int64            `view:"-" desc:"the current random seed"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &deep.Network{}
	ss.NDepthCode = 8
	ss.TrnEpcLog = &etable.Table{}
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
	ss.LayStatNms = []string{"MT", "MTD", "SMA", "SMAD"}
	ss.PosAFNms = []string{"SMA", "SMAD", "7a", "7aD"}
	ss.NewPrjns()
}

// NewPrjns creates new projections
func (ss *Sim) NewPrjns() {
	ss.Topo4Prjn = prjn.NewPoolTile()
	ss.Topo4Prjn.Size.Set(4, 4)
	ss.Topo4Prjn.Skip.Set(2, 2)
	ss.Topo4Prjn.Start.Set(-1, -1)
	ss.Topo4PrjnRecip = prjn.NewPoolTile()
	*ss.Topo4PrjnRecip = *ss.Topo4Prjn
	ss.Topo4PrjnRecip.Recip = true
	ss.Topo3Prjn = prjn.NewPoolTile()
	ss.Topo3Prjn.Size.Set(3, 3)
	ss.Topo3Prjn.Skip.Set(1, 1)
	ss.Topo3Prjn.Start.Set(-1, -1)
	ss.Topo4S4Prjn = prjn.NewPoolTile()
	ss.Topo4S4Prjn.Size.Set(4, 4)
	ss.Topo4S4Prjn.Skip.Set(4, 4)
	ss.Topo4S4Prjn.Start.Set(0, 0)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
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

	ss.TrainEnv.Defaults()
	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Event.Max = 100
	ss.TrainEnv.Run.Max = ss.MaxRuns
	ss.TrainEnv.PosRes = 32                  // higher res
	ss.TrainEnv.PosPop.Sigma.Set(0.02, 0.02) // tighter for 32 -- .05 for 16
	ss.TrainEnv.Init(0)
	ss.TrainEnv.Validate()

	ss.PopCode.Defaults()
	ss.PopCode.Min = 0.1 // key -- never gets much below .25
	ss.PopCode.Max = 1.1
}

func (ss *Sim) ConfigNet(net *deep.Network) {
	net.InitName(net, "MapNav")
	v2, v2p := net.AddInputPulv4D("V2d", 8, 16, 1, ss.NDepthCode)

	mt, mtd, _ := net.AddSuperDeep4D("MT", 4, 8, 6, 6, false, false) // no pulv, attn
	mtctxt := mtd.RecvPrjns().SendName("MT")
	mtctxt.SetPattern(ss.Topo3Prjn)
	mt.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "V2d", XAlign: relpos.Left, YAlign: relpos.Front})
	mtd.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "MT", YAlign: relpos.Front, Space: 2})

	sma, smad, _ := net.AddSuperDeep2D("SMA", 8, 8, false, false) // no pulv, attn
	sma.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "MTD", YAlign: relpos.Front})
	smad.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "SMA", YAlign: relpos.Front, Space: 2})

	ipl, ipld, _ := net.AddSuperDeep4D("7a", 2, 4, 8, 8, false, false) // no pulv, attn
	ipl.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "SMA", XAlign: relpos.Left})
	ipld.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "7a", YAlign: relpos.Front, Space: 2})

	m1, m1p := net.AddInputPulv2D("M1", 1, int(navenv.ActionsN))
	m1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "SMAD", YAlign: relpos.Front, Space: 4, YOffset: 2})

	v2.SetClass("V2d")
	v2p.SetClass("V2d")

	mt.SetClass("MT")
	mtd.SetClass("MT")

	ipl.SetClass("7a")
	ipld.SetClass("7a")

	sma.SetClass("SMA")
	smad.SetClass("SMA")

	net.ConnectLayers(v2, mt, ss.Topo4Prjn, emer.Forward)
	net.ConnectLayers(mtd, v2p, ss.Topo4PrjnRecip, emer.Forward)
	net.ConnectLayers(v2p, mtd, ss.Topo4Prjn, emer.Back)
	net.ConnectLayers(v2p, mt, ss.Topo4Prjn, emer.Back)

	net.ConnectLayers(mt, ipl, ss.Topo4Prjn, emer.Forward)
	net.ConnectLayers(ipl, mt, ss.Topo4PrjnRecip, emer.Back)
	net.ConnectLayers(ipld, v2p, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(v2p, ipld, ss.Topo4S4Prjn, emer.Back)
	net.ConnectLayers(v2p, ipl, ss.Topo4S4Prjn, emer.Back)

	net.ConnectLayers(ipl, sma, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(sma, ipl, prjn.NewFull(), emer.Back)
	net.ConnectLayers(smad, v2p, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(smad, ipld, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(v2p, smad, prjn.NewFull(), emer.Back)
	net.ConnectLayers(v2p, sma, prjn.NewFull(), emer.Back)

	net.ConnectLayers(sma, m1, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(smad, m1p, prjn.NewFull(), emer.Forward)

	net.ConnectLayers(smad, mtd, prjn.NewFull(), emer.Back)

	net.ConnectLayers(m1p, smad, prjn.NewFull(), emer.Back)
	net.ConnectLayers(m1p, sma, prjn.NewFull(), emer.Back)

	ss.TRCLays = make([]string, 0, 10)
	nl := ss.Net.NLayers()
	for li := 0; li < nl; li++ {
		ly := ss.Net.Layer(li)
		if ly.Type() == deep.TRC {
			ss.TRCLays = append(ss.TRCLays, ly.Name())
		}
	}

	// using 4 total threads -- todo: didn't work
	mt.SetThread(1)
	mtd.SetThread(1)
	ipl.SetThread(2)
	ipld.SetThread(2)
	sma.SetThread(3)
	smad.SetThread(3)

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
	v2 := net.LayerByName("V2d")
	mt := net.LayerByName("MT")
	mtd := net.LayerByName("MTD")
	ipl := net.LayerByName("7a")
	ipld := net.LayerByName("7aD")
	v2mt := mt.RecvPrjns().SendName("V2d").(*deep.Prjn)
	scales := &etensor.Float32{}
	ss.Topo4Prjn.TopoWts(v2.Shape(), mt.Shape(), scales)
	v2mt.SetScalesRPool(scales)

	v2pmt := mt.RecvPrjns().SendName("V2dP").(*deep.Prjn)
	ss.Topo4Prjn.TopoWts(v2.Shape(), mt.Shape(), scales)
	v2pmt.SetScalesRPool(scales)

	v2pmtd := mtd.RecvPrjns().SendName("V2dP").(*deep.Prjn)
	ss.Topo4Prjn.TopoWts(v2.Shape(), mtd.Shape(), scales)
	v2pmtd.SetScalesRPool(scales)

	v2pipl := ipl.RecvPrjns().SendName("V2dP").(*deep.Prjn)
	ss.Topo4Prjn.TopoWts(v2.Shape(), ipl.Shape(), scales)
	v2pipl.SetScalesRPool(scales)

	v2pipld := ipld.RecvPrjns().SendName("V2dP").(*deep.Prjn)
	ss.Topo4Prjn.TopoWts(v2.Shape(), ipld.Shape(), scales)
	v2pipld.SetScalesRPool(scales)

	net.InitWts()
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

	m1 := net.LayerByName("M1").(*deep.Layer)
	m1.UnitVals(&ss.ActVals, "ActM") // action generated last time

	_, mi := norm.MaxIdx32(ss.ActVals)
	netact := navenv.Actions(mi)

	ss.TrainEnv.SetAction(netact)
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	v2d := net.LayerByName("V2d").(*deep.Layer)
	depth := en.State("Depth")
	ny := depth.Dim(0)
	nx := depth.Dim(1)
	for y := 0; y < ny; y++ {
		for x := 0; x < nx; x++ {
			d := depth.FloatVal([]int{y, x}) // not flipped
			dxf := (d - .2) / .8
			if dxf > 1 {
				dxf = 1
			}
			if dxf < 0 {
				dxf = 0
			}
			ss.PopCode.Encode(&ss.PopVals, float32(d), ss.NDepthCode)
			si := v2d.Shp.Offset([]int{y, x, 0, 0})
			for i := 0; i < ss.NDepthCode; i++ {
				nrn := &v2d.Neurons[si+i]
				if nrn.IsOff() {
					continue
				}
				nrn.Ext = ss.PopVals[i]
			}
		}
	}
	v2d.UpdateExtFlags()

	pats := en.State("ActMap")
	m1.ApplyExt(pats)

	ss.CurPosMap = en.State("PosMap").(*etensor.Float32)
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
	nt := len(ss.TRCLays)
	if len(ss.TrlCosDiffTRC) != nt {
		ss.TrlCosDiffTRC = make([]float64, nt)
		ss.SumCosDiffTRC = make([]float64, nt)
		ss.EpcCosDiffTRC = make([]float64, nt)
	}
	acd := 0.0
	for i, ln := range ss.TRCLays {
		ly := ss.Net.LayerByName(ln).(*deep.Layer)
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
	for i := range ss.TRCLays {
		ss.EpcCosDiffTRC[i] = ss.SumCosDiffTRC[i] / nt
		ss.SumCosDiffTRC[i] = 0
		acd += ss.EpcCosDiffTRC[i]
	}
	ss.EpcCosDiff = acd / 3
}

// UpdtPosAFs updates position activation rf's
func (ss *Sim) UpdtPosAFs() {
	if ss.PosAFTsr == nil {
		ss.PosAFTsr = &etensor.Float32{}
	}
	naf := len(ss.PosAFNms)
	if len(ss.PosAFs.RFs) != naf {
		for _, lnm := range ss.PosAFNms {
			ly := ss.Net.LayerByName(lnm)
			if ly == nil {
				continue
			}
			ly.UnitValsTensor(ss.PosAFTsr, "ActM")
			af := ss.PosAFs.AddRF(lnm, ss.PosAFTsr, ss.CurPosMap)
			af.NormRF.SetMetaData("min", "0")
			af.NormRF.SetMetaData("colormap", "JetMuted")
		}
	}
	for _, lnm := range ss.PosAFNms {
		ly := ss.Net.LayerByName(lnm)
		if ly == nil {
			continue
		}
		ly.UnitValsTensor(ss.PosAFTsr, "ActM")
		ss.PosAFs.Add(lnm, ss.PosAFTsr, ss.CurPosMap, 0.01) // thr prevent weird artifacts
	}
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	// inp := ss.Net.LayerByName("V2dP").(*deep.Layer)
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
	ss.UpdtPosAFs()
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

	for i, lnm := range ss.TRCLays {
		dt.SetCellFloat(lnm+" CosDiff", row, float64(ss.EpcCosDiffTRC[i]))
	}

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(*deep.Layer)
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab, true)
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
	for _, lnm := range ss.TRCLays {
		sch = append(sch, etable.Column{lnm + " CosDiff", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "MapNav Epoch Plot"
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

	for _, lnm := range ss.TRCLays {
		plt.SetColParams(lnm+" CosDiff", true, true, 0, true, 1)
	}
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", false, true, 0, true, .5)
	}

	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	// inp := ss.Net.LayerByName("InputP").(*deep.Layer)

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
		ly := ss.Net.LayerByName(lnm).(*deep.Layer)
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}
	// if ss.InputValsTsr == nil { // re-use same tensors so not always reallocating mem
	// 	ss.InputValsTsr = &etensor.Float32{}
	// 	ss.OutputValsTsr = &etensor.Float32{}
	// }
	// inp.UnitValsTensor(ss.InputValsTsr, "ActM")
	// dt.SetCellTensor("InActM", row, ss.InputValsTsr)
	// inp.UnitValsTensor(ss.InputValsTsr, "ActP")
	// dt.SetCellTensor("InActP", row, ss.OutputValsTsr)
	// trg.UnitValsTensor(ss.OutputValsTsr, "ActP")
	// dt.SetCellTensor("Targs", row, ss.OutputValsTsr)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	// inp := ss.Net.LayerByName("InputP").(*deep.Layer)

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
	plt.Params.Title = "MapNav Test Trial Plot"
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
	plt.Params.Title = "MapNav Testing Epoch Plot"
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
		ly := ss.Net.LayerByName(lnm).(*deep.Layer)
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
	plt.Params.Title = "MapNav Test Cycle Plot"
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
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab, true)
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
	plt.Params.Title = "MapNav Run Plot"
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

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("MapNav")
	gi.SetAppAbout(`This learns 2D map (grid cells) by moving around a map. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewWindow2D("MapNav", "Map Navigation Predictive Learning", width, height, true)
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
		ewin := TheSim.TrainEnv.OpenWindow()
		ewin.GoStartEventLoop()
		win.StartEventLoop()
	}
}
