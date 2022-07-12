// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// emery2 is a simulated virtual rat / cat, using axon spiking model
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/decoder"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// Debug triggers various messages etc
var Debug = false

func main() {
	TheSim.New()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Config()
	TheSim.Init()
	win := TheSim.ConfigGui()
	ev := TheSim.Envs[etime.Train.String()].(*FWorld)
	fwin := ev.ConfigWorldGui()
	fwin.GoStartEventLoop()
	win.StartEventLoop()
}

// see params_def.go for default params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *deep.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	PctCortex    float64          `desc:"proportion of action driven by the cortex vs. hard-coded reflexive subcortical"`
	PctCortexMax float64          `desc:"maximum PctCortex, when running on the schedule"`
	Prjns        Prjns            `desc:"special projections"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Time         axon.Time        `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`
	PCAInterval  int              `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	MaxTrls      int              `desc:"maximum number of training trials per epoch"`
	Decoder      decoder.SoftMax  `desc:"decoder for better output"`
	NOutPer      int              `desc:"number of units per localist output unit"`
	SubPools     bool             `desc:"if true, organize layers and connectivity with 2x2 sub-pools within each topological pool"`
	RndOutPats   bool             `desc:"if true, use random output patterns -- else localist"`
	ConfusionEpc int              `desc:"epoch to start recording confusion matrix"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
	Comm     *mpi.Comm   `view:"-" desc:"mpi communicator"`
	AllDWts  []float32   `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts  []float32   `view:"-" desc:"buffer of MPI summed dwt weight changes"`
	RFTargs  []string    `view:"-" desc:"special targets for activation-based receptive field maps"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for args when calling methods
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &deep.Network{}
	ss.Prjns.New()
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.PctCortexMax = 0.9
	ss.NOutPer = 5
	ss.SubPools = true
	ss.RndOutPats = false
	ss.PCAInterval = 10
	ss.ConfusionEpc = 500
	ss.MaxTrls = 512
	ss.RFTargs = []string{"Pos", "Act", "Ang", "Rot"}
	ss.Time.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *FWorld
	if len(ss.Envs) == 0 {
		trn = &FWorld{}
		tst = &FWorld{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*FWorld)
		tst = ss.Envs.ByMode(etime.Test).(*FWorld)
	}

	trn.Config(ss.MaxTrls)
	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	trn.Init(0)
	trn.Validate()

	tst.Config(ss.MaxTrls)
	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	tst.Init(0)
	tst.Validate()

	// todo!
	// if ss.Args.Bool("mpi") {
	// 	if Debug {
	// 		mpi.Printf("Did Env MPIAlloc\n")
	// 	}
	// 	trn.MPIAlloc()
	// 	tst.MPIAlloc()
	// }

	ss.Envs.Add(trn, tst)
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

	ev := ss.Envs[etime.Train.String()].(*FWorld)
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
	mstdct.RecvPrjns().SendName(mstd.Name()).SetPattern(p1to1)                                      // todo: try ss.Prjn3x3Skp1 orig: p1to1
	net.ConnectLayers(mstdct, v2wdp, ss.Prjns.Prjn4x4Skp2Recip, emer.Forward).SetClass("CTToPulv2") // 3 is too high
	net.ConnectLayers(v2wdp, mstd, ss.Prjns.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(v2wdp, mstdct, ss.Prjns.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	// net.ConnectCtxtToCT(v2wd, mstdct, ss.Prjns.Prjn4x4Skp2).SetClass("CTFmSuper")
	// net.ConnectCtxtToCT(mstdct, mstdct, parprjn).SetClass("CTSelf") // important!

	mstd.SetRepIdxsShape(emer.CenterPoolIdxs(mstd, 2), emer.CenterPoolShape(mstd, 2))
	mstdct.SetRepIdxsShape(emer.CenterPoolIdxs(mstdct, 2), emer.CenterPoolShape(mstdct, 2))

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
	net.ConnectLayers(lipct, v2fdp, ss.Prjns.Prjn4x3Skp2Recip, emer.Forward).SetClass("CTToPulv3")
	net.ConnectLayers(v2fdp, lipct, ss.Prjns.Prjn4x3Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(v2fdp, lip, ss.Prjns.Prjn4x3Skp2, emer.Back).SetClass("FmPulv")
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

	net.ConnectLayers(v2wd, mstd, ss.Prjns.Prjn4x4Skp2, emer.Forward).SetClass("SuperFwd")

	// MStd <-> CIPl
	net.ConnectLayers(mstd, cipl, ss.Prjns.Prjn4x4Skp2, emer.Forward).SetClass("SuperFwd")
	net.ConnectLayers(cipl, mstd, ss.Prjns.Prjn4x4Skp2Recip, emer.Back)
	net.ConnectLayers(ciplct, mstdct, ss.Prjns.Prjn4x4Skp2Recip, emer.Back).SetClass("CTBack")

	net.ConnectLayers(mstdp, ciplct, ss.Prjns.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(mstdp, cipl, ss.Prjns.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(ciplct, mstdp, ss.Prjns.Prjn4x4Skp2Recip, emer.Forward).SetClass("CTToPulv3")

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
	net.ConnectLayers(v2fd, lip, ss.Prjns.Prjn4x3Skp2, emer.Forward)

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

	// net.ConnectLayers(mstdct, ciplp, ss.Prjns.Prjn4x4Skp2, emer.Forward).SetClass("FwdToPulv")

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

	v2wd.SetThread(1)
	v2wdp.SetThread(1)
	mstd.SetThread(1)
	mstdct.SetThread(1)
	mstdp.SetThread(1)

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	if !ss.Args.Bool("nogui") {
		sr := net.SizeReport()
		mpi.Printf("%s", sr)
	}
	ar := net.ThreadReport() // hand tuning now..
	mpi.Printf("%s", ar)
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.InitRndSeed()
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.Params.SetAll()
	ss.Net.SlowInterval = 100 // 100 > 20
	ss.NewRun()
	ss.ViewUpdt.Update()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	effTrls := ss.MaxTrls
	if ss.Args.Bool("mpi") {
		effTrls /= mpi.WorldSize() // todo: use more robust fun
		if Debug {
			mpi.Printf("MPI trials: %d\n", effTrls)
		}
	}

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, effTrls).AddTime(etime.Cycle, 200)

	// note: needs a lot of data for good actrfs -- 100 here
	man.AddStack(etime.Test).AddTime(etime.Epoch, 100).AddTime(etime.Trial, effTrls).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Time, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Time, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Replace("UpdateWeights", func() {
		ss.Net.DWt(&ss.Time)
		ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible
		ss.MPIWtFmDWt()
	})

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("Env:Step", func() {
			// note: OnStart for env.Env, others may happen OnEnd
			ss.Envs[mode.String()].Step()
		})
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
		stack.Loops[etime.Trial].OnEnd.Add("StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("TrialStats", ss.TrialStats)
	}

	// note: plusPhase is shared between all stacks!
	plusPhase, _ := man.Stacks[etime.Train].Loops[etime.Cycle].EventByName("PlusPhase")
	plusPhase.OnEvent.Add("TakeAction", func() {
		ss.TakeAction(ss.Net)
	})

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("RandCheck", func() {
	// 	if ss.Args.Bool("mpi") {
	// 		empi.RandCheck(ss.Comm) // prints error message
	// 	}
	// })

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		axon.LogTestErrors(&ss.Logs)
	})
	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.PCAInterval > 0) && (trnEpc%ss.PCAInterval == 0) {
			if ss.Args.Bool("mpi") {
				ss.Logs.MPIGatherTableRows(etime.Analyze, etime.Trial, ss.Comm)
			}
			axon.PCAStats(ss.Net.AsAxon(), &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.PCAInterval > 0) && (trnEpc%ss.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// this is actually fairly expensive
	man.GetLoop(etime.Test, etime.Trial).OnEnd.Add("ActRFs", func() {
		ss.UpdateActRFs()
		ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01)
	})
	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("CheckEpc", func() {
		if ss.Args.Bool("actrfs") {
			trnEpc := man.Stacks[etime.Test].Loops[etime.Epoch].Counter.Cur
			fmt.Printf("epoch: %d\n", trnEpc)
		}
	})

	// Save weights to file at end, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveActRFs", func() {
		if ss.Args.Bool("actrfs") {
			ss.TestAll()
			ss.SaveAllActRFs()
		}
	})

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PctCortex", func() {
		trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if trnEpc > 1 && trnEpc%5 == 0 {
			ss.PctCortex = float64(trnEpc) / 50
			if ss.PctCortex > ss.PctCortexMax {
				ss.PctCortex = ss.PctCortexMax
			} else {
				fmt.Printf("PctCortex updated to: %g at epoch: %d\n", ss.PctCortex, trnEpc)
			}
		}
	})

	////////////////////////////////////////////
	// GUI
	if ss.Args.Bool("nogui") {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt)
		axon.LooperUpdtPlots(man, &ss.GUI)

		man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("UpdateWorldGui", func() {
			ev := ss.Envs[etime.Train.String()].(*FWorld)
			ev.UpdateWorldGui()
		})
	}

	if Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// SaveWeights saves weights with filename recording run, epoch
func (ss *Sim) SaveWeights() {
	ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
	axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) TakeAction(net *deep.Network) {
	ev := ss.Envs[ss.Time.Mode].(*FWorld)
	nact := ss.DecodeAct(ev)
	gact, urgency := ev.ActGen()
	ss.Stats.SetString("NetAction", ev.Acts[nact])
	ss.Stats.SetString("GenAction", ev.Acts[gact])
	if nact == gact {
		ss.Stats.SetFloat("ActMatch", 1)
	} else {
		ss.Stats.SetFloat("ActMatch", 0)
	}
	actAct := ss.Stats.String("GenAction")
	if erand.BoolProb(float64(urgency), -1) {
		actAct = ss.Stats.String("GenAction")
	} else if erand.BoolProb(ss.PctCortex, -1) {
		actAct = ss.Stats.String("NetAction")
	}
	ss.Stats.SetString("ActAction", actAct)

	ly := net.LayerByName("VL").(axon.AxonLayer).AsAxon()
	ly.SetType(emer.Input)
	ev.Action(actAct, nil)
	ap, ok := ev.Pats[actAct]
	if ok {
		ly.ApplyExt(ap)
	}
	ly.SetType(emer.Target)
	// fmt.Printf("action: %s\n", ev.Acts[act])
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *FWorld) int {
	vt := ss.Stats.SetLayerTensor(ss.Net, "VL", "ActM")
	act := ev.DecodeAct(vt)
	return act
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	ev := ss.Envs[ss.Time.Mode].(*FWorld)

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
		pats := ev.State(states[i])
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	trnev := ss.Envs[etime.Train.String()].(*FWorld)
	trnev.Init(0)
	trnev.InitPos(mpi.WorldRank()) // start in diff locations for mpi nodes
	tstev := ss.Envs[etime.Test.String()].(*FWorld)
	tstev.Init(0)
	tstev.InitPos(mpi.WorldRank())
	ss.Time.Reset()
	ss.Time.Mode = etime.Train.String()
	ss.Net.InitWts()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Stats.ActRFs.Reset()
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
	ss.Stats.ActRFsAvgNorm()
	ss.GUI.ViewActRFs(&ss.Stats.ActRFs)
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	// ss.Logs.ResetLog(etime.Test, etime.Epoch) // only show last row
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.GUI.Stopped()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	var mode etime.Modes
	mode.FromString(ss.Time.Mode)
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ss.Stats.SetFloat("PctCortex", ss.PctCortex)
	ev := ss.Envs[ss.Time.Mode].(*FWorld)
	ss.Stats.SetString("TrialName", ev.String())
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "NetAction", "GenAction", "ActAction", "ActMatch", "Cycle", "TrlCorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("VL").(axon.AxonLayer).AsAxon()

	ss.Stats.SetFloat("TrlCorSim", float64(out.CorSim.Cor))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	// note: most stats computed in TakeAction
	ss.Stats.SetFloat("TrlErr", 1-ss.Stats.Float("ActMatch"))

	// epc := env.Epoch.Cur
	// if epc > ss.ConfusionEpc {
	// 	ss.Stats.Confusion.Incr(ss.Stats.Int("TrlCatIdx"), ss.Stats.Int("TrlRespIdx"))
	// }
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "PctCortex")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "NetAction", "GenAction", "ActAction")

	// todo: make basic err stats actually useful

	ss.Logs.AddStatAggItem("CorSim", "TrlCorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", "TrlUnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ActMatch", "ActMatch", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	// Copy over Testing items
	// ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr")

	deep.LogAddTRCCorSimItems(&ss.Logs, ss.Net, etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigActRFs()

	axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)

	// todo: PCA items should apply to CT layers too -- pass a type here.
	axon.LogAddPCAItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "Target")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.AllModes, etime.Cycle, "Target")

	ss.Logs.PlotItems("PctCortex", "ActMatch", "Energy", "Hydra", "V2WdP_CorSim")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net.AsAxon())
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
}

// ConfigLogItems specifies extra logging items
func (ss *Sim) ConfigLogItems() {
	ev := TheSim.Envs[etime.Train.String()].(*FWorld)
	ss.Logs.AddItem(&elog.Item{
		Name:      "ActCor",
		Type:      etensor.FLOAT64,
		CellShape: []int{len(ev.Acts)},
		DimNames:  []string{"Acts"},
		Plot:      true,
		Range:     minmax.F64{Min: 0},
		TensorIdx: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IdxView(ctx.Mode, etime.Trial)
				spl := split.GroupBy(ix, []string{"GenAction"})
				split.AggTry(spl, "ActMatch", agg.AggMean)
				ags := spl.AggsToTable(etable.ColNameOnly)
				ss.Logs.MiscTables["ActCor"] = ags
				ctx.SetTensor(ags.Cols[0]) // cors
			}}})
	for _, nm := range ev.Acts { // per-action % correct
		anm := nm // closure
		ss.Logs.AddItem(&elog.Item{
			Name:  anm + "Cor",
			Type:  etensor.FLOAT64,
			Plot:  true,
			Range: minmax.F64{Min: 0},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ags := ss.Logs.MiscTables["ActCor"]
					rw := ags.RowsByString("GenAction", anm, etable.Equals, etable.UseCase)
					if len(rw) > 0 {
						ctx.SetFloat64(ags.CellFloat("ActMatch", rw[0]))
					}
				}}})
	}
	for _, nm := range ev.Inters { // interoceptive internal variable state
		inm := nm
		ss.Logs.AddItem(&elog.Item{
			Name:  inm,
			Type:  etensor.FLOAT64,
			Plot:  true,
			Range: minmax.F64{Min: 0},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					ctx.SetFloat32(ev.InterStates[ctx.Item.Name])
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
	}

}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Time.Mode = mode.String() // Also set specifically in a Loop callback.
	}
	ss.StatCounters()

	if ss.Args.Bool("mpi") && time == etime.Epoch { // Must gather data for trial level if doing epoch level
		ss.Logs.MPIGatherTableRows(mode, etime.Trial, ss.Comm)
	}

	dt := ss.Logs.Table(mode, time)
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc

	// trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	// if trnEpc > ss.ConfusionEpc {
	// 	ss.Stats.Confusion.Probs()
	// 	fnm := ecmd.LogFileName("trn_conf", ss.Net.Name(), ss.Stats.String("RunName"))
	// 	ss.Stats.Confusion.SaveCSV(gi.FileName(fnm))
	// }
}

// ConfigActRFs
func (ss *Sim) ConfigActRFs() {
	ev := ss.Envs[etime.Train.String()].(*FWorld)
	for _, trg := range ss.RFTargs {
		mt := ss.Stats.F32Tensor(trg)
		switch trg {
		case "Pos":
			mt.CopyShapeFrom(ev.World)
		case "Act":
			mt.SetShape([]int{len(ev.Acts)}, nil, nil)
		case "Ang":
			mt.SetShape([]int{ev.NRotAngles}, nil, nil)
		case "Rot":
			mt.SetShape([]int{3}, nil, nil)
		}
	}

	lnms := []string{"MSTd", "cIPL", "PCC", "SMA"}
	var arfs []string
	for _, lnm := range lnms {
		for _, trg := range ss.RFTargs {
			arfs = append(arfs, lnm+":"+trg)
		}
	}
	ss.Stats.InitActRFs(ss.Net, arfs, "ActM")
}

// UpdateActRFs updates position activation rf's
func (ss *Sim) UpdateActRFs() {
	ev := ss.Envs[ss.Time.Mode].(*FWorld)
	for _, trg := range ss.RFTargs {
		mt := ss.Stats.F32Tensor(trg)
		mt.SetZeros()
		switch trg {
		case "Pos":
			mt.Set([]int{ev.PosI.Y, ev.PosI.X}, 1)
		case "Act":
			mt.Set1D(ev.Act, 1)
		case "Ang":
			mt.Set1D(ev.Angle/15, 1)
		case "Rot":
			mt.Set1D(1+ev.RotAng/15, 1)
		}
	}
}

// SaveAllActRFs saves all ActRFs to files, using LogFileName from ecmd using RunName
func (ss *Sim) SaveAllActRFs() {
	ss.Stats.ActRFsAvgNorm()
	for _, paf := range ss.Stats.ActRFs.RFs {
		fnm := ecmd.LogFileName(paf.Name, "ActRF", ss.Stats.String("RunName"))
		etensor.SaveCSV(&paf.NormRF, gi.FileName(fnm), '\t')
	}
}

// OpenAllActRFs open all ActRFs from directory of given path
func (ss *Sim) OpenAllActRFs(path gi.FileName) {
	ss.UpdateActRFs()
	ss.Stats.ActRFsAvgNorm()
	ap := string(path)
	if strings.HasSuffix(ap, ".tsv") {
		ap, _ = filepath.Split(ap)
	}
	vp := ss.GUI.Win.Viewport
	for _, paf := range ss.Stats.ActRFs.RFs {
		fnm := ecmd.LogFileName(paf.Name, "ActRF", ss.Stats.String("RunName")) // todo: won't work for other runs
		ffnm := filepath.Join(ap, fnm)
		err := etensor.OpenCSV(&paf.NormRF, gi.FileName(ffnm), '\t')
		if err != nil {
			fmt.Println(err)
		} else {
			etview.TensorGridDialog(vp, &paf.NormRF, giv.DlgOpts{Title: "Act RF " + paf.Name, Prompt: paf.Name, TmpSave: nil}, nil, nil)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	cam := &(nv.Scene().Camera)
	cam.Pose.Pos.Set(0, 2.2, 1.9)
	cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Emery"
	ss.GUI.MakeWindow(ss, "lvis", title, `Full brain predictive learning in navigational / survival environment. See <a href="https://github.com/ccnlab/map-nav/blob/master/sims/emery2/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdt = &ss.ViewUpdt
	ss.ConfigNetView(nv)

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddActRFGridTabs(&ss.Stats.ActRFs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train, etime.Test})

	ss.GUI.ToolBar.AddSeparator("test")

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset ActRFs",
		Icon:    "reset",
		Tooltip: "reset current position activation rfs accumulation data",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Stats.ActRFs.Reset()
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "View ActRFs",
		Icon:    "file-image",
		Tooltip: "compute activation rfs and view them.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Stats.ActRFsAvgNorm()
			for _, paf := range ss.Stats.ActRFs.RFs {
				etview.TensorGridDialog(ss.GUI.ViewPort, &paf.NormRF, giv.DlgOpts{Title: "Act RF " + paf.Name, Prompt: paf.Name, TmpSave: nil}, nil, nil)
			}
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Open ActRFs",
		Icon:    "file-open",
		Tooltip: "Open saved ARF .tsv files -- select a path or specific file in path",
		Active:  egui.ActiveStopped,
		Func: func() {
			giv.CallMethod(ss, "OpenAllActRFs", ss.GUI.ViewPort)
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test All",
		Icon:    "step-fwd",
		Tooltip: "Tests a large same of testing items and records ActRFs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.RunTestAll()
			}
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset TrialLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated trial log",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Trial)
			ss.GUI.UpdatePlot(etime.Train, etime.Trial)
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "New Seed",
		Icon:    "new",
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RndSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/map-nav/blob/main/sims/emery2/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.AddInt("iticycles", 0, "number of cycles to run between trials (inter-trial-interval)")
	ss.Args.SetInt("epochs", 2000)
	ss.Args.SetInt("runs", 1)
	ss.Args.AddBool("mpi", false, "if set, use MPI for distributed computation")
	ss.Args.AddBool("actrfs", false, "if true, save final activation-based rf's after each run")
	ss.Args.Parse() // always parse
}

// These props register methods so they can be called through gui with arg prompts
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"OpenAllActRFs", ki.Props{
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
	if ss.Args.Bool("mpi") {
		ss.MPIInit()
	}

	// key for Config and Init to be after MPIInit
	ss.Config()
	ss.Init()

	ss.Args.ProcStd(&ss.Params)

	if mpi.WorldRank() == 0 {
		ss.Args.ProcStdLogs(&ss.Logs, &ss.Params, ss.Net.Name())
	}

	ss.Args.SetBool("nogui", true)                                       // by definition if here
	ss.Stats.SetString("RunName", ss.Params.RunName(ss.Args.Int("run"))) // used for naming logs, stats, etc

	if mpi.WorldRank() != 0 {
		ss.Args.SetBool("wts", false)
	}

	netdata := ss.Args.Bool("netdata")
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	mpi.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs

	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")

	ss.NewRun()
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.MPIFinalize()
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
		ss.Args.SetBool("mpi", false)
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.Args.Bool("mpi") {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
// includes all other long adapting factors too: DTrgAvg, ActAvg, etc
func (ss *Sim) CollectDWts(net *axon.Network) {
	net.CollectDWts(&ss.AllDWts)
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	if ss.Args.Bool("mpi") {
		ss.CollectDWts(ss.Net.AsAxon())
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ss.SumDWts, mpi.WorldSize())
	}
	ss.Net.WtFmDWt(&ss.Time)
}
