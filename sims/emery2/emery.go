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
	"runtime"
	"strings"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/decoder"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/prjn"
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

var (
	// Debug triggers various messages etc
	Debug = false
	// GPU runs with the GPU (for demo, testing -- not useful for such a small network)
	GPU = false
)

func main() {
	sim := &Sim{}
	sim.New()
	if len(os.Args) > 1 {
		sim.RunNoGUI() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			sim.RunGUI()
		})
	}
}

// see params_def.go for default params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	PctCortex    float64          `desc:"proportion of action driven by the cortex vs. hard-coded reflexive subcortical"`
	PctCortexMax float64          `desc:"maximum PctCortex, when running on the schedule"`
	Prjns        Prjns            `desc:"special projections"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Context      axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`
	PCAInterval  int              `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	MaxTrls      int              `desc:"maximum number of training trials per epoch"`
	Decoder      decoder.SoftMax  `desc:"decoder for better output"`
	NOutPer      int              `desc:"number of units per localist output unit"`
	SubPools     bool             `desc:"if true, organize layers and connectivity with 2x2 sub-pools within each topological pool"`
	RndOutPats   bool             `desc:"if true, use random output patterns -- else localist"`
	ConfusionEpc int              `desc:"epoch to start recording confusion matrix"`
	SubStep      int

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

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Prjns.New()
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.PctCortexMax = 0.3
	ss.NOutPer = 5
	ss.SubPools = true
	ss.RndOutPats = false
	ss.PCAInterval = 10
	ss.ConfusionEpc = 500
	ss.MaxTrls = 1024
	ss.RFTargs = []string{"Pos", "Act", "HdDir"}
	ss.Context.Defaults()
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

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Emery")

	full := prjn.NewFull()
	sameu := prjn.NewPoolSameUnit()
	sameu.SelfCon = false
	p1to1 := prjn.NewPoolOneToOne()
	_ = p1to1
	one2one := prjn.NewOneToOne()
	_ = one2one

	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1
	_ = rndcut

	var ff, fb emer.Prjn

	nPerAng := 5 // 5*1 impairs M1P prediction somewhat -- nothing else
	nPerDepth := 1
	rfDepth := 6
	rfWidth := 3
	stdHidX := 6 // smaller 10 -> 6 impairs M1P prediction somewhat, not actmatch etc
	stdHidY := 6

	rect := prjn.NewRect()
	rect.Size.Set(rfWidth, rfDepth) // 6 > 8 > smaller
	rect.Scale.Set(1.0/float32(nPerAng), 1.0/float32(nPerDepth))
	_ = rect

	ev := ss.Envs[etime.Train.String()].(*FWorld)
	fsz := 1 + 2*ev.FoveaSize
	// popsize = 12

	space := float32(2)

	////////////////////////////////////////
	// input / output layers:

	v2wd, v2wdP := net.AddInputPulv4D("V2Wd", 1, ev.NFOVRays, ev.DepthSize, 1, 2*space)
	v2wd.SetClass("Depth")
	v2wdP.SetClass("Depth")

	// skip fovea depth for now:
	// v2fd, v2fdp := net.AddInputPulv4D("V2Fd", ev.DepthPools, fsz, ev.DepthSize/ev.DepthPools, 1, space) // FovDepth
	// v2fd.SetClass("Depth")
	// v2fdp.SetClass("Depth")

	// v1f, v1fP := net.AddInputPulv4D("V1F", 1, fsz, ev.PatSize.Y, ev.PatSize.X, space) // Fovea
	v1f := net.AddLayer4D("V1F", 1, fsz, ev.PatSize.Y, ev.PatSize.X, axon.InputLayer) // Fovea
	v1f.SetClass("Fovea")
	// v1fP.SetClass("Fovea")

	// s1s, s1sP := net.AddInputPulv4D("S1S", 1, 4, 2, 1, space) // ProxSoma
	s1s := net.AddLayer4D("S1S", 1, 4, 2, 1, axon.InputLayer)
	s1s.SetClass("S1S")
	// s1sP.SetClass("S1S")

	hd, hdP := net.AddInputPulv2D("HeadDir", 1, ev.PopSize, space)
	hd.SetClass("HeadDir")
	hdP.SetClass("HeadDir")

	ins := net.AddLayer4D("Ins", 1, len(ev.Inters), ev.PopSize, 1, axon.InputLayer) // Inters = Insula
	ins.SetClass("Ins")

	act := net.AddLayer2D("Act", ev.PatSize.Y, ev.PatSize.X, axon.InputLayer) // Action: what is actually done
	act.SetClass("Action")
	vl := net.AddPulvLayer2D("VL", ev.PatSize.Y, ev.PatSize.X) // VL predicts brainstem Action
	vl.SetBuildConfig("DriveLayName", act.Name())
	vl.SetClass("Action")

	m1, m1CT := net.AddSuperCT2D("M1", "M1Prjn", stdHidY, stdHidX, space, one2one)
	m1P := net.AddPulvForSuper(m1, space)

	// vl is a predictive thalamus but we don't have direct access to its source
	net.ConnectToPulv(m1, m1CT, vl, full, full, "")

	ff, _ = net.BidirConnectLayers(m1, vl, full)
	ff.SetClass("StrongFF")

	//////////////////////////////////
	// Hidden layers

	// vestibular / head direction hidden layer:
	s2v, s2vCT := net.AddSuperCT2D("S2V", "S2V", stdHidY, stdHidX, space, one2one)
	s2vCT.SetClass("S2V CTCopy")
	// net.ConnectCTSelf(s2vCT, full) // not for one-step
	net.ConnectToPulv(s2v, s2vCT, hdP, full, full, "")
	net.ConnectLayers(hd, s2v, full, axon.ForwardPrjn)

	// it, itCT := net.AddSuperCT2D("IT", "IT", stdHidY, stdHidX, space, one2one)
	// itCT.SetClass("IT CTCopy")
	// net.ConnectCTSelf(itCT, full)
	// net.ConnectToPulv(it, itCT, v1fP, full, full, "")
	// net.ConnectLayers(v1f, it, full, axon.ForwardPrjn).SetClass("SuperFwd")

	// lip, lipCT := net.AddSuperCT4D("LIP", "LIP", ev.DepthPools/2, 1, 8, 8, space, one2one)
	// lipCT.SetClass("LIP CTCopy")
	// net.ConnectCTSelf(lipCT, full)
	// net.ConnectToPulv(lip, lipCT, v2fdp, ss.Prjns.Prjn4x3Skp2Recip, ss.Prjns.Prjn4x3Skp2, "")
	// net.ConnectLayers(v2fd, lip, ss.Prjns.Prjn4x3Skp2, axon.ForwardPrjn).SetClass("SuperFwd")

	// todo: LIP fovea is not topo organized for left, middle right positions
	// groups are depth organized
	// not enough resolution to really map that out here.
	// net.ConnectLayers(lipCT, v1fP, full, axon.BackPrjn).SetClass("ToPulv1") // attention

	// 4D topo MSTd with pools:
	// // mstd, mstdCT, mstdP := net.AddSuperCTPulv4D("MSTd", "MSTd", ev.DepthPools/2, ev.NFOVRays/2, 8, 8, space, one2one) // was p1to1
	// mstd, mstdCT := net.AddSuperCT4D("MSTd", ev.DepthPools/2, ev.NFOVRays/2, 8, 8, space, one2one) // was p1to1
	// // todo: try ss.Prjn3x3Skp1 orig: p1to1
	// net.ConnectCTSelf(mstdCT, p1to1)
	// net.ConnectToPulv(mstd, mstdCT, v2wdP, ss.Prjns.Prjn4x4Skp2Recip, ss.Prjns.Prjn4x4Skp2, "")
	// // net.ConnectCtxtToCT(v2wd, mstdCT, ss.Prjns.Prjn4x4Skp2).SetClass("CTFmSuper")
	// mstd.SetRepIdxsShape(emer.CenterPoolIdxs(mstd, 2), emer.CenterPoolShape(mstd, 2))
	// mstdCT.SetRepIdxsShape(emer.CenterPoolIdxs(mstdCT, 2), emer.CenterPoolShape(mstdCT, 2))

	// 2D MSTd -- much simpler:
	mstdSz := evec.Vec2i{X: (ev.NFOVRays - (rfWidth - 1)) * nPerAng, Y: (ev.DepthSize - (rfDepth - 1)) * nPerDepth}
	mstd, mstdCT := net.AddSuperCT2D("MSTd", "MSTd", mstdSz.Y, mstdSz.X, space, one2one) // def one2one
	mstdCT.SetClass("MSTd CTCopy")
	// mstdP.SetClass("MSTd")
	// net.ConnectCTSelf(mstdCT, full) // not needed or beneficial for simple 1 step move pred
	net.ConnectToPulv(mstd, mstdCT, v2wdP, full, rect, "")
	net.ConnectLayers(v2wd, mstd, rect, axon.ForwardPrjn)

	// note: skipping cipl for now:
	// cipl, ciplCT := net.AddSuperCT4D("cIPL", "cIPL", 3, 3, 8, 8, space, one2one)
	// // cipl, ciplCT, ciplP := net.AddSuperCTPulv4D("cIPL", 3, 3, 8, 8, space, one2one)
	// ciplCT.SetClass("cIPL")
	// ciplP.SetClass("cIPL")
	// net.ConnectCTSelf(ciplCT, p1to1)
	// net.ConnectToPulv(cipl, ciplCT, v2wdP, full, full, "")
	// v2wdP.RecvPrjns().SendName(ciplCT.Name()).SetClass("ToPulv1")
	// MStd <-> CIPl
	// net.ConnectLayers(mstd, cipl, ss.Prjns.Prjn4x4Skp2, axon.ForwardPrjn).SetClass("SuperFwd")
	// net.ConnectLayers(cipl, mstd, ss.Prjns.Prjn4x4Skp2Recip, axon.BackPrjn)
	// net.ConnectLayers(ciplCT, mstdCT, ss.Prjns.Prjn4x4Skp2Recip, axon.BackPrjn).SetClass("CTBack")
	// net.ConnectLayers(mstdP, ciplCT, ss.Prjns.Prjn4x4Skp2, axon.BackPrjn).SetClass("FmPulv")
	// net.ConnectLayers(mstdP, cipl, ss.Prjns.Prjn4x4Skp2, axon.BackPrjn).SetClass("FmPulv")
	// net.ConnectLayers(ciplCT, mstdP, ss.Prjns.Prjn4x4Skp2Recip, axon.ForwardPrjn).SetClass("CTToPulv3")
	// net.ConnectLayers(sma, cipl, full, axon.BackPrjn)
	// net.ConnectLayers(s1s, cipl, full, axon.BackPrjn)
	// net.ConnectLayers(s2v, cipl, full, axon.BackPrjn)
	// net.ConnectLayers(vl, cipl, full, axon.BackPrjn) // todo: m1?
	// net.ConnectLayers(pccct, ciplCT, full, axon.BackPrjn).SetClass("CTBack")
	// net.ConnectLayers(smact, ciplCT, full, axon.BackPrjn).SetClass("CTBack")
	// net.ConnectLayers(mstdCT, ciplP, ss.Prjns.Prjn4x4Skp2, axon.ForwardPrjn).SetClass("FwdToPulv")

	// 4D topo pcc:
	// pcc, pccCT := net.AddSuperCT4D("PCC", "PCC", 2, 2, 7, 7, space, one2one)
	// net.ConnectCTSelf(pccCT, p1to1)
	// net.ConnectToPulv(pcc, pccCT, v2wdP, full, full, "")
	// v2wdP.RecvPrjns().SendName(pccCT.Name()).SetClass("ToPulv1")

	// 2d PCC -- full layer consolidating MST space
	// bidir connected to MSTDd super, contributes prediction to V2WdP
	// pcc can use a slightly larger layer -- not critical but it is a central hub..
	pcc, pccCT := net.AddSuperCT2D("PCC", "PCC", 8, 8, space, one2one)
	pccCT.SetClass("PCC CTInteg")
	ff, _ = pccCT.SendNameTry(pcc.Name())
	ff.SetClass("FixedCTFmSuper")
	net.ConnectCTSelf(pccCT, full, "")                   // longer time integration to integrate depth map..
	net.ConnectToPulv(pcc, pccCT, v2wdP, full, full, "") // top-down depth pred
	ff, _ = net.BidirConnectLayers(mstd, pcc, full)
	ff.SetClass("StrongFF") // needs extra activity
	// net.ConnectLayers(mstd, pcc, full, axon.ForwardPrjn)
	// net.ConnectLayers(pccCT, mstdCT, full, axon.BackPrjn).SetClass("CTBack")

	// pcc integrates somatosensory as well
	net.ConnectLayers(s1s, pcc, full, axon.ForwardPrjn)
	// net.BidirConnectLayers(s2v, pcc, full) // todo: just direct head dir input instead?
	net.ConnectLayers(s2v, pcc, full, axon.ForwardPrjn)

	// sma gets from everything, predicts m1P
	sma, smaCT := net.AddSuperCT2D("SMA", "SMA", stdHidY, stdHidX, space, one2one)
	smaCT.SetClass("SMA CTCopy") // CTCopy seems better than CTInteg
	// net.ConnectCTSelf(smaCT, full, "") // maybe not
	net.ConnectToPulv(sma, smaCT, m1P, full, full, "")
	ff, _ = net.BidirConnectLayers(pcc, sma, full)
	ff.SetClass("StrongFF") // needs extra activity
	// net.ConnectLayers(smaCT, pccCT, full, axon.BackPrjn).SetClass("CTBack") // not useful
	ff, _ = net.BidirConnectLayers(sma, m1, full)
	ff.SetClass("StrongFF")

	// net.ConnectLayers(vl, sma, full, axon.BackPrjn) // get the error directly?
	// net.ConnectLayers(vl, smaCT, full, axon.BackPrjn)

	net.ConnectLayers(v1f, sma, full, axon.ForwardPrjn)
	net.ConnectLayers(s1s, sma, full, axon.ForwardPrjn)

	_, fb = net.BidirConnectLayers(s2v, sma, full)
	fb.SetClass("StrongFmSMA")

	// net.ConnectLayers(smaCT, vl, full, axon.ForwardPrjn) // this may be key?
	// net.ConnectLayers(sma, vl, full, axon.ForwardPrjn) // no, right?

	net.ConnectLayers(sma, mstd, full, axon.BackPrjn).SetClass("StrongFmSMA") // note: back seems to work?
	// net.ConnectLayers(smaCT, mstdCT, full, axon.BackPrjn).SetClass("CTBack") // not useful

	// net.ConnectLayers(it, sma, full, axon.ForwardPrjn)
	// net.ConnectLayers(lip, sma, full, axon.ForwardPrjn)
	// net.ConnectLayers(smaCT, s2vCT, full, axon.BackPrjn).SetClass("CTBack")

	//////////////////////////////////////
	// Action prediction

	// from act works perfectly if pctcortex is 0 -- important sanity check
	// on the depth, head direction predictive learning
	// net.ConnectLayers(act, mstd, full, axon.ForwardPrjn).SetClass("FmAct")
	// net.ConnectLayers(act, s2v, full, axon.ForwardPrjn).SetClass("FmAct")
	// net.ConnectLayers(act, pcc, full, axon.ForwardPrjn).SetClass("FmAct")
	// net.ConnectLayers(act, sma, full, axon.ForwardPrjn).SetClass("FmAct")
	// net.ConnectLayers(act, cipl, full, axon.ForwardPrjn).SetClass("FmAct")
	// net.ConnectLayers(act, it, full, axon.ForwardPrjn).SetClass("FmAct")
	// net.ConnectLayers(act, lip, full, axon.ForwardPrjn).SetClass("FmAct")

	////////////////////
	// lateral inhibition

	/*
		net.LateralConnectLayerPrjn(mstd, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
		net.LateralConnectLayerPrjn(mstdCT, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
		// net.LateralConnectLayerPrjn(cipl, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
		// net.LateralConnectLayerPrjn(ciplCT, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
		net.LateralConnectLayerPrjn(pcc, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
		net.LateralConnectLayerPrjn(pccCT, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
		net.LateralConnectLayerPrjn(sma, full, &axon.HebbPrjn{}).SetType(emer.Inhib)
		net.LateralConnectLayerPrjn(smaCT, full, &axon.HebbPrjn{}).SetType(emer.Inhib)
		net.LateralConnectLayerPrjn(m1, full, &axon.HebbPrjn{}).SetType(emer.Inhib)
	*/

	////////////////////
	// Shortcuts

	// net.ConnectLayers(v2wd, cipl, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	// net.ConnectLayers(v2wd, ciplCT, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	// net.ConnectLayers(v2wd, pcc, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	// net.ConnectLayers(v2wd, pccCT, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	// net.ConnectLayers(v2wd, sma, rndcut, axon.ForwardPrjn).SetClass("V1SC")
	// net.ConnectLayers(v2wd, smaCT, rndcut, axon.ForwardPrjn).SetClass("V1SC")

	//////////////////////////////////////
	// position

	v1f.PlaceRightOf(v2wdP, space)
	// v2fd.PlaceBehind(v1f, space)
	v2wdP.PlaceRightOf(v2wd, space)

	// v1fP.PlaceBehind(v1f, space)
	// it.PlaceBehind(v1fP, space)
	// itCT.PlaceBehind(it, space)

	// v2fd.PlaceRightOf(v1f, space)
	// v2fdp.PlaceBehind(v2fd, space)
	// lip.PlaceRightOf(v2fdp, space)
	// lipCT.PlaceRightOf(lip, space)

	s1s.PlaceBehind(v1f, space)
	// s1sP.PlaceBehind(s1s, space)

	hd.PlaceBehind(s1s, space)
	hdP.PlaceBehind(hd, space)

	s2v.PlaceBehind(hdP, space)
	s2vCT.PlaceRightOf(s2v, space)

	ins.PlaceRightOf(v1f, 2*space)

	vl.PlaceRightOf(ins, space)
	act.PlaceBehind(vl, space)

	mstd.PlaceAbove(v2wd)

	// cipl.PlaceRightOf(mstd, space)

	pcc.PlaceRightOf(mstd, space)

	sma.PlaceRightOf(pcc, space)

	m1.PlaceRightOf(sma, space)

	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.Params.SetObject("Network")

	// if !ss.Args.Bool("nogui") {
	// 	sr := net.SizeReport()
	// 	mpi.Printf("%s", sr)
	// }
	net.InitWts()

	thr := ss.Args.Int("threads")
	runtime.GOMAXPROCS(thr)
	net.SetNThreads(thr)
	mpi.Printf("GOMAXPROCS: %d\tthreads: %d\n", runtime.GOMAXPROCS(0), net.NThreads)
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
	ss.ViewUpdt.RecordSyns()
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

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 150).AddTime(etime.Trial, effTrls).AddTime(etime.Cycle, 200)

	// note: needs a lot of data for good actrfs -- is relatively fast under mpi
	man.AddStack(etime.Test).AddTime(etime.Epoch, 200).AddTime(etime.Trial, effTrls).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Replace("UpdateWeights", func() {
		ss.Net.DWt(&ss.Context)
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
			axon.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
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
	// man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("CheckEpc", func() {
	// 	if ss.Args.Bool("actrfs") {
	// 		trnEpc := man.Stacks[etime.Test].Loops[etime.Epoch].Counter.Cur
	// 		fmt.Printf("epoch: %d\n", trnEpc)
	// 	}
	// })

	// Save weights to file at end, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveActRFs", func() {
		if ss.Args.Bool("actrfs") {
			ss.TestAll()
		}
	})

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PctCortex", func() {
		trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if trnEpc > 50 && trnEpc%5 == 0 {
			ss.PctCortex = float64(trnEpc-50) / 50
			if ss.PctCortex > ss.PctCortexMax {
				ss.PctCortex = ss.PctCortexMax
			} else {
				mpi.Printf("PctCortex updated to: %g at epoch: %d\n", ss.PctCortex, trnEpc)
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
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net)
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
	if mpi.WorldRank() > 0 {
		return
	}
	ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
	axon.SaveWeightsIfArgSet(ss.Net, &ss.Args, ctrString, ss.Stats.String("RunName"))
}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) TakeAction(net *axon.Network) {
	if ss.SubStep == 0 {
		ss.SubStep++
		return
	}
	// fmt.Printf("Take Action\n")
	ss.SubStep = 0 // reset for next time
	ev := ss.Envs[ss.Context.Mode.String()].(*FWorld)
	nact := ss.DecodeAct(ev)
	gact, urgency := ev.InstinctAct()
	ss.Stats.SetString("PrevAction", ss.Stats.String("ActAction"))
	ss.Stats.SetString("NetAction", ev.Acts[nact])
	ss.Stats.SetString("Instinct", ev.Acts[gact])
	if nact == gact {
		ss.Stats.SetFloat("ActMatch", 1)
	} else {
		ss.Stats.SetFloat("ActMatch", 0)
	}
	actAct := ss.Stats.String("Instinct")
	if erand.BoolP32(urgency, -1) {
		actAct = ss.Stats.String("Instinct")
	} else if erand.BoolP(ss.PctCortex, -1) {
		actAct = ss.Stats.String("NetAction")
	}
	ss.Stats.SetString("ActAction", actAct)

	ev.Action(actAct, nil)
	// ap, ok := ev.Pats[actAct]
	// if ok {
	// 	ss.Net.ApplyExts(&ss.Context)
	// }
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
	ev := ss.Envs[ss.Context.Mode.String()].(*FWorld)

	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "HeadDir", "Inters", "Action"}
	lays := []string{"V2Wd", "V2Fd", "V1F", "S1S", "HeadDir", "Ins", "Act"}
	for i, lnm := range lays {
		ly := ss.Net.AxonLayerByName(lnm)
		if ly == nil {
			continue
		}
		pats := ev.State(states[i])
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
	net.ApplyExts(&ss.Context)
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
	ss.Context.Reset()
	ss.Context.Mode = etime.Train
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
	if ss.GUI.Active {
		ss.Stats.ActRFsAvgNorm()
		ss.GUI.ViewActRFs(&ss.Stats.ActRFs)
		// ss.SaveAllActRFs() // test
	} else {
		ss.SaveAllActRFs()
	}
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
	ss.Stats.SetFloat("ActMatch", 0.0)
	ss.Stats.SetString("PrevAction", "")
	ss.Stats.SetString("NetAction", "")
	ss.Stats.SetString("Instinct", "")
	ss.Stats.SetString("ActAction", "")
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ss.Stats.SetFloat("PctCortex", ss.PctCortex)
	ev := ss.Envs[ss.Context.Mode.String()].(*FWorld)
	ss.Stats.SetString("TrialName", ev.String())
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "PrevAction", "NetAction", "Instinct", "ActAction", "ActMatch", "Cycle", "TrlCorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.AxonLayerByName("VL")
	v2wdP := ss.Net.AxonLayerByName("V2WdP")

	ss.Stats.SetFloat("TrlCorSim", float64(v2wdP.Vals.CorSim.Cor))
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
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "NetAction", "Instinct", "ActAction")

	// todo: make basic err stats actually useful

	ss.Logs.AddStatAggItem("CorSim", "TrlCorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", "TrlUnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ActMatch", "ActMatch", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	// Copy over Testing items
	// ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr")

	axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigActRFs()

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)

	axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net, etime.Test, etime.Cycle)
	// ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")
	// ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.AllModes, etime.Cycle, "TargetLayer")

	ss.Logs.PlotItems("PctCortex", "ActMatch", "Energy", "Hydra", "V2WdP_CorSim")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
}

// ConfigLogItems specifies extra logging items
func (ss *Sim) ConfigLogItems() {
	ev := ss.Envs[etime.Train.String()].(*FWorld)
	ss.Logs.AddItem(&elog.Item{
		Name:      "ActCor",
		Type:      etensor.FLOAT64,
		CellShape: []int{len(ev.Acts)},
		DimNames:  []string{"Acts"},
		Plot:      false,
		Range:     minmax.F64{Min: 0},
		TensorIdx: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IdxView(ctx.Mode, etime.Trial)
				spl := split.GroupBy(ix, []string{"Instinct"})
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
					rw := ags.RowsByString("Instinct", anm, etable.Equals, etable.UseCase)
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
	if mode != etime.Analyze {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
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
		case "HdDir":
			mt.SetShape([]int{ev.NMotAngles}, nil, nil)
		}
	}

	lnms := []string{"MSTd", "PCC", "PCCCT", "SMA"} //  "cIPL",
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
	ev := ss.Envs[ss.Context.Mode.String()].(*FWorld)
	for _, trg := range ss.RFTargs {
		mt := ss.Stats.F32Tensor(trg)
		mt.SetZeros()
		switch trg {
		case "Pos":
			mt.Set([]int{ev.PosI.Y, ev.PosI.X}, 1)
		case "Act":
			mt.Set1D(ev.Act, 1)
		case "HdDir":
			mt.Set1D(ev.HeadDir/ev.MotAngInc, 1)
			// case "Rot":
			// 	mt.Set1D(1+ev.RotAng/ev.NMotAngles, 1)
		}
	}
}

// SaveAllActRFs saves all ActRFs to files, using LogFileName from ecmd using RunName
func (ss *Sim) SaveAllActRFs() {
	if mpi.WorldSize() > 1 {
		ss.Stats.ActRFs.MPISum(ss.Comm)
		if mpi.WorldRank() > 0 {
			return // don't save!
		}
	}
	ss.Stats.ActRFsAvgNorm()
	for _, paf := range ss.Stats.ActRFs.RFs {
		fnm := ecmd.LogFileName(paf.Name, "ActRF", ss.Stats.String("RunName"))
		etensor.SaveCSV(&paf.NormRF, gi.FileName(fnm), '\t')
	}
	for _, trg := range ss.RFTargs {
		paf := ss.Stats.ActRFs.RFByName("SMA:" + trg)
		fnm := ecmd.LogFileName(trg, "ActRFSrc", ss.Stats.String("RunName"))
		etensor.SaveCSV(&paf.NormSrc, gi.FileName(fnm), '\t')
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
			fmt.Printf("ffnm: %s\n", ffnm)
			fmt.Println(err)
		} else {
			etview.TensorGridDialog(vp, &paf.NormRF, giv.DlgOpts{Title: "Act RF " + paf.Name, Prompt: paf.Name, TmpSave: nil}, nil, nil)
		}
	}
	for _, trg := range ss.RFTargs {
		paf := ss.Stats.ActRFs.RFByName("SMA:" + trg)
		fnm := ecmd.LogFileName(trg, "ActRFSrc", ss.Stats.String("RunName"))
		ffnm := filepath.Join(ap, fnm)
		err := etensor.OpenCSV(&paf.NormSrc, gi.FileName(ffnm), '\t')
		if err != nil {
			fmt.Printf("ffnm: %s\n", ffnm)
			fmt.Println(err)
		} else {
			etview.TensorGridDialog(vp, &paf.NormSrc, giv.DlgOpts{Title: "ActSrc RF " + paf.Name, Prompt: paf.Name, TmpSave: nil}, nil, nil)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	cam := &(nv.Scene().Camera)
	cam.Pose.Pos.Set(0, 1.9, 2.2)
	cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Emery"
	ss.GUI.MakeWindow(ss, "emery2", title, `Full brain predictive learning in navigational / survival environment. See <a href="https://github.com/ccnlab/map-nav/blob/master/sims/emery2/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.03
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
	if GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win
}

func (ss *Sim) RunGUI() {
	ss.Config()
	ss.Init()
	win := ss.ConfigGui()
	ev := ss.Envs[etime.Train.String()].(*FWorld)
	fwin := ev.ConfigWorldGui()
	fwin.GoStartEventLoop()
	win.StartEventLoop()
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.AddInt("threads", 1, "number of threads per node")
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

func (ss *Sim) RunNoGUI() {
	if ss.Args.Bool("mpi") {
		ss.MPIInit()
	}

	// key for Config and Init to be after MPIInit
	ss.Config()

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

	ss.Init()

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	mpi.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs
	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")

	gpu := ss.Args.Bool("gpu")
	if gpu {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}

	ss.NewRun()
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy()
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
		ss.CollectDWts(ss.Net)
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ss.SumDWts, mpi.WorldSize())
	}
	ss.Net.WtFmDWt(&ss.Context)
}
