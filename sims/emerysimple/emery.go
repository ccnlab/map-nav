// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// emery2 is a simulated virtual rat / cat, using axon spiking model
package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/actrf"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/pca"
	"log"
	"math/rand"
)

func main() {
	TheSim.New()    // note: not running Config here -- done in CmdArgs for mpi / nogui
	TheSim.Config() // for GUI case, config then run..
	TheSim.Init()
	TheSim.Train()
}

// see params_def.go for default params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct { // TODO(refactor): Remove a lot of this stuff
	Net *deep.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`

	PctCortex        float64        `desc:"proportion of action driven by the cortex vs. hard-coded reflexive subcortical"`
	PctCortexMax     float64        `desc:"maximum PctCortex, when running on the schedule"`
	ARFs             actrf.RFs      `view:"no-inline" desc:"activation-based receptive fields"`
	TrnErrStats      *etable.Table  `view:"no-inline" desc:"stats on train trials where errors were made"`
	TrnAggStats      *etable.Table  `view:"no-inline" desc:"stats on all train trials"`
	RunStats         *etable.Table  `view:"no-inline" desc:"aggregate stats on all runs"`
	MinusCycles      int            `desc:"number of minus-phase cycles"`
	PlusCycles       int            `desc:"number of plus-phase cycles"`
	ErrLrMod         axon.LrateMod  `view:"inline" desc:"learning rate modulation as function of error"`
	Params           params.Sets    `view:"no-inline" desc:"full collection of param sets"`
	ParamSet         string         `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag              string         `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Prjn4x4Skp2      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x4Skp2Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn4x3Skp2      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x3Skp2Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn3x3Skp1      *prjn.PoolTile `view:"no-inline" desc:"feedforward 3x3 skip 1 topo prjn"`
	Prjn4x4Skp4      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn"`
	Prjn4x4Skp4Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn, recip"`
	MaxRuns          int            `desc:"maximum number of model runs to perform"`
	MaxEpcs          int            `desc:"maximum number of epochs to run per model run"`
	TestEpcs         int            `desc:"number of epochs of testing to run, cumulative after MaxEpcs of training"`
	RepsInterval     int            `desc:"how often to analyze the representations"`
	TrainEnv         FWorld         `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	OnlyEnv          DWorld         `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time             axon.Time      `desc:"axon timing parameters and state"`
	TrainUpdt        etime.Times    `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt         etime.Times    `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval     int            `desc:"how often to run through all the test patterns, in terms of training epochs"`
	CosDifActs       []string       `view:"-" desc:"actions to track CosDif performance by"`
	InitOffNms       []string       `desc:"names of layers to turn off initially"`
	LayStatNms       []string       `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	ARFLayers        []string       `desc:"names of layers to compute position activation fields on"`

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
	PopVals      []float32                   `view:"-" desc:"tmp pop code values"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	SaveARFs     bool                        `view:"-" desc:"for command-line run only, auto-save receptive field data"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	RndSeed      int64                       `view:"-" desc:"the current random seed"`
	UseMPI       bool                        `view:"-" desc:"if true, use MPI to distribute computation across nodes"`
	SaveProcLog  bool                        `view:"-" desc:"if true, save logs per processor"`
	Comm         *mpi.Comm                   `view:"-" desc:"mpi communicator"`
	AllDWts      []float32                   `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts      []float32                   `view:"-" desc:"buffer of MPI summed dwt weight changes"`

	// Characteristics of the environment interface.
	FoveaSize  int        `desc:"number of items on each size of the fovea, in addition to center (0 or more)"`
	DepthSize  int        `inactive:"+" desc:"number of units in depth population codes"`
	DepthPools int        `inactive:"+" desc:"number of pools to divide DepthSize into"`
	PatSize    evec.Vec2i `desc:"size of patterns for mats, acts"`
	Inters     []string   `desc:"list of interoceptive body states, represented as pop codes"`
	PopSize    int        `inactive:"+" desc:"number of units in population codes"`
	NFOVRays   int        `inactive:"+" desc:"total number of FOV rays that are traced"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() { // TODO(refactor): Remove a lot
	ss.Net = &deep.Network{}

	ss.RunStats = &etable.Table{}

	ss.Time.Defaults()
	ss.MinusCycles = 150
	ss.PlusCycles = 50
	ss.RepsInterval = 10

	ss.ErrLrMod.Defaults()
	ss.ErrLrMod.Base = 0.05 // 0.05 >= .01, .1 -- hard to tell
	ss.ErrLrMod.Range.Set(0.2, 0.8)
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.TrainUpdt = etime.AlphaCycle
	ss.TestUpdt = etime.GammaCycle
	ss.CosDifActs = []string{"Forward", "Left", "Right"}
	ss.InitOffNms = []string{"MSTdP", "cIPLCT", "cIPLP", "PCCCT"}
	ss.LayStatNms = []string{"MSTd", "MSTdCT", "SMA", "SMACT"}
	ss.ARFLayers = []string{"MSTd", "cIPL", "PCC", "SMA"}

	// Default values
	ss.PctCortexMax = 0.9 // 0.5 before
	ss.TestInterval = 50000

	ss.NewPrjns()
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
	//ss.ConfigEnv()// TODO(DWORLD!)
	ss.ConfigNet(ss.Net)
}

//func (ss *Sim) ConfigEnv() {// TODO(DWORLD!)
//	ss.TrainEnv.Config(200) // 1000) // n trials per epoch
//	ss.TrainEnv.Nm = "TrainEnv"
//	ss.TrainEnv.Dsc = "training params and state"
//	ss.OnlyEnv.GetCounter(etime.Run).Max = ss.MaxRuns
//	ss.TrainEnv.Init(0)
//	ss.TrainEnv.Validate()
//
//	ss.ConfigRFMaps()
//}

func (ss *Sim) ConfigNet(net *deep.Network) {
	net.InitName(net, "Emery")

	// DO NOT SUBMIT Not sure these should go here.// TODO(DWORLD!)
	ss.DepthPools = 8
	ss.DepthSize = 32
	ss.NFOVRays = 13 // Not sure this value is right
	ss.FoveaSize = 1
	ss.PopSize = 16
	ss.Inters = []string{"Energy", "Hydra", "BumpPain", "FoodRew", "WaterRew"}
	ss.PatSize = evec.Vec2i{X: 5, Y: 5}

	full := prjn.NewFull()
	sameu := prjn.NewPoolSameUnit()
	sameu.SelfCon = false
	p1to1 := prjn.NewPoolOneToOne()

	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1
	_ = rndcut

	var parprjn prjn.Pattern
	parprjn = full

	fsz := 1 + 2*ss.FoveaSize
	// popsize = 12

	// input / output layers:
	v2wd, v2wdp := net.AddInputTRC4D("V2Wd", ss.DepthPools, ss.NFOVRays, ss.DepthSize/ss.DepthPools, 1)
	v2fd, v2fdp := net.AddInputTRC4D("V2Fd", ss.DepthPools, fsz, ss.DepthSize/ss.DepthPools, 1) // FovDepth
	v2wd.SetClass("Depth")
	v2wdp.SetClass("Depth")
	v2fd.SetClass("Depth")
	v2fdp.SetClass("Depth")

	v1f, v1fp := net.AddInputTRC4D("V1F", 1, fsz, ss.PatSize.Y, ss.PatSize.X) // Fovea
	v1f.SetClass("Fovea")
	v1fp.SetClass("Fovea")

	s1s, s1sp := net.AddInputTRC4D("S1S", 1, 4, 2, 1) // ProxSoma
	s1s.SetClass("S1S")
	s1sp.SetClass("S1S")

	s1v, s1vp := net.AddInputTRC4D("S1V", 1, 2, ss.PopSize, 1) // Vestibular
	s1v.SetClass("S1V")
	s1vp.SetClass("S1V")

	ins := net.AddLayer4D("Ins", 1, len(ss.Inters), ss.PopSize, 1, emer.Input) // Inters = Insula
	ins.SetClass("Ins")

	m1 := net.AddLayer2D("M1", 10, 10, emer.Hidden)
	vl := net.AddLayer2D("VL", ss.PatSize.Y, ss.PatSize.X, emer.Target)  // Action
	act := net.AddLayer2D("Act", ss.PatSize.Y, ss.PatSize.X, emer.Input) // Action

	m1p := net.AddTRCLayer2D("M1P", 10, 10)
	m1p.Driver = "M1"

	mstd, mstdct, mstdp := net.AddSuperCTTRC4D("MSTd", ss.DepthPools/2, ss.NFOVRays/2, 8, 8)
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

	lip, lipct := net.AddSuperCT4D("LIP", ss.DepthPools/2, 1, 8, 8)
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

	net.ConnectLayers(smact, itct, parprjn, emer.Back).SetClass("CTBack") // needs to know how moving.
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
	// position // TODO(refactor): This GUI stuff should maybe be separated? at least it should be commented as GUI

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
	// collect // TODO(refactor): this is used for like logging and stuff?

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

	net.Defaults()                                                             // TODO(refactor): why isn't all this in a function?
	SetParams("Network", ss.LogSetParams, ss.Net, &ss.Params, ss.ParamSet, ss) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() { // TODO(refactor): this should be broken up
	rand.Seed(ss.RndSeed)
	//ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc // TODO(DWORLD!)
	SetParams("", ss.LogSetParams, ss.Net, &ss.Params, ss.ParamSet, ss) // all sheets
	ss.NewRun()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion.
func (ss *Sim) Counters(train bool) string { // TODO(refactor): GUI
	// if train {
	return "It's 5 o'clock somewhere!" // TODO(DWORLD!)
	//return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tEvent:\t%d\tCycle:\t%d\tAct:\t%v\tNet:\t%v\t\t\t", ss.OnlyEnv.GetCounter(etime.Run).Cur, ss.OnlyEnv.GetCounter(etime.Epoch).Cur, ss.TrainEnv.Event.Cur, ss.Time.Cycle, ss.ActAction, ss.NetAction)
	// } else {
	// 	return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tEvent:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.OnlyEnv.GetCounter(etime.Run).Cur, ss.OnlyEnv.GetCounter(etime.Epoch).Cur, ss.TestEnv.Event.Cur, ss.Time.Cycle, ss.TrainEnv.Event.Cur)
	// }
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up.. // TODO(refactor): All this goes to looper library code

// ThetaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of ThetaCycle
func (ss *Sim) ThetaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine

	mode := etime.Train.String()
	if !train {
		mode = etime.Test.String()
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt(&ss.Time)
	} else {
		ss.Net.SynFail(&ss.Time)
	}

	ev := &ss.TrainEnv

	minusCyc := ss.MinusCycles
	plusCyc := ss.PlusCycles

	ss.Net.NewState()
	ss.Time.NewState(mode)

	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)

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

	}
	ss.Time.NewPhase(true)
	ss.TakeAction(ss.Net, ev) // TODO(refactor): this seems different

	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)

		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}

	}

	ss.TrialStats(train) // need stats for lrmod

	if train {
		ss.Net.DWt(&ss.Time)
	}

}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) TakeAction(net *deep.Network, ev *FWorld) { // TODO(refactor): call this in looper
	ly := net.LayerByName("VL").(axon.AxonLayer).AsAxon()
	nact := ss.DecodeAct(ly, ev)
	//gact, urgency := ev.ActGen() // TODO(DWORLD!)
	urgency := 0
	gact := 0
	ss.NetAction = "Right" // ev.Acts[nact]
	ss.GenAction = "Right" // ev.Acts[gact]
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

// DecodeAct decodes the VL ActM state to find the closest action pattern
func (ss *Sim) DecodeAct(ly *axon.Layer, ev *FWorld) int { //where should this go
	vt := ValsTsr(&ss.ValsTsrs, "VL")
	ly.UnitValsTensor(vt, "ActM")
	//act := ev.DecodeAct(vt) // TODO(DWORLD!)
	return 0
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() { // TODO(refactor): looper code
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {

		ss.EpochSched(epc)
		ss.TrainEnv.Event.Cur = 0

		if epc >= ss.MaxEpcs {
			if ss.SaveWts { // doing this earlier
				SaveWeights(WeightsFileName(ss.Net.Nm, ss.Tag, ss.ParamSet, ss.OnlyEnv.GetCounter(etime.Run).Cur, ss.OnlyEnv.GetCounter(etime.Epoch).Cur), ss.Net)
			}
			// done with training..
			if ss.SaveARFs {
				ss.TestAll()
			}
			ss.RunEnd()
			run := ss.OnlyEnv.GetCounter(etime.Run)
			run.Incr()
			_, _, chg = run.Query()
			if chg { // we are done!
				return
			} else {
				return
			}
		}
	}
	states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "Vestibular", "Inters", "Action", "Action"}
	layers := []string{"V2Wd", "V2Fd", "V1F", "S1S", "S1V", "Ins", "VL", "Act"}
	ApplyInputs(ss.Net, &ss.TrainEnv, states, layers)
	ss.ThetaCyc(true) // train
	// ss.TrialStats(true) // now in alphacyc

	if ss.RepsInterval > 0 && epc%ss.RepsInterval == 0 {

	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc. here
func (ss *Sim) RunEnd() { // TODO(refactor): looper call

	// if ss.SaveWts { // doing this earlier
	// 	ss.SaveWeights()
	// }
	if ss.SaveARFs {
		ss.SaveAllARFs()
	}
}

// NewRun intializes a new run of the model, using the OnlyEnv.GetCounter(etime.Run) counter
// for the new run value
func (ss *Sim) NewRun() { // TODO(refactor): looper call
	//run := ss.OnlyEnv.GetCounter(etime.Run).Cur
	ss.PctCortex = 0
	//ss.TrainEnv.Init(run) // TODO(DWORLD!)
	// ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() { // TODO(refactor): use Stats
	// accumulators
	ss.NumTrlStats = 0
	ss.SumActMatch = 0
	ss.SumCosDiff = 0
	// clear rest just to make Sim look initialized
	ss.EpcActMatch = 0
	ss.EpcCosDiff = 0
}

// TrialStatsTRC computes the trial-level statistics for TRC layers
func (ss *Sim) TrialStatsTRC(accum bool) { // TODO(refactor): looper stats?
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

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) { // TODO(refactor): looper call
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
func (ss *Sim) TrainEpoch() { // TODO(refactor): replace with looper
	curEpc := ss.OnlyEnv.GetCounter(etime.Epoch).Cur
	for {
		ss.TrainTrial()
		if ss.OnlyEnv.GetCounter(etime.Epoch).Cur != curEpc {
			break
		}
	}

}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() { // TODO(refactor): replace with looper
	curRun := ss.OnlyEnv.GetCounter(etime.Run).Cur
	for {
		ss.TrainTrial()
		if ss.OnlyEnv.GetCounter(etime.Run).Cur != curRun {
			break
		}
	}

}

// EpochSched implements the learning rate schedule etc.
func (ss *Sim) EpochSched(epc int) { // TODO(refactor): looper callback
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
func (ss *Sim) Train() { // TODO(refactor): delete, looper
	for {
		ss.TrainTrial()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// note: using TrainEnv for everything

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) { // TODO(refactor): replace with looper
	ss.OnlyEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epoch := ss.OnlyEnv.GetCounter(etime.Epoch)
	epc, _, chg := epoch.Query()
	if chg {

		ss.EpochSched(epc)
		ss.TrainEnv.Event.Cur = 0

		if epc >= ss.TestEpcs {
			return
		}
	}

	states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "Vestibular", "Inters", "Action", "Action"}
	layers := []string{"V2Wd", "V2Fd", "V1F", "S1S", "S1V", "Ins", "VL", "Act"}
	ApplyInputs(ss.Net, &ss.TrainEnv, states, layers)
	ss.ThetaCyc(false) // train
	// ss.TrialStats(true) // now in alphacyc

}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() { // TODO(refactor): replace with looper
	curRun := ss.OnlyEnv.GetCounter(etime.Run).Cur
	for {
		ss.TestTrial(false)
		if ss.OnlyEnv.GetCounter(etime.Run).Cur != curRun {
			break
		}
	}

}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() { // TODO(refactor): delete, looper
	ss.TestAll()
}

//////////////////////////////////////////////
//  TrnTrlRepLog

// CenterPoolsIdxs returns the indexes for 2x2 center pools (including sub-pools):
// nu = number of units per pool, sis = starting indexes
func (ss *Sim) CenterPoolsIdxs(ly *axon.Layer) (nu int, sis []int) { // TODO(refactor): Axon code?
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
func (ss *Sim) CopyCenterPools(ly *axon.Layer, vl *etensor.Float32) { // TODO(refactor): axon code?
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
