// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// eboa is a simulated virtual rat / cat, using axon spiking model,
// with the bg, ofc, acc (BOA) decision making system.
package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/decoder"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/econfig"
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
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/timer"
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
	"github.com/goki/ki/bools"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		gimain.Main(sim.RunGUI)
	} else {
		sim.RunNoGUI()
	}
}

// see params_def.go for default params, config.go for Config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Config    Config           `desc:"simulation configuration parameters -- set by .toml config file and / or args"`
	Net       *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	StopOnSeq bool             `desc:"if true, stop running at end of a sequence (for NetView Di data parallel index)"`
	StopOnErr bool             `desc:"if true, stop running when an error programmed into the code occurs"`
	Params    emer.NetParams   `view:"inline" desc:"all parameter management"`
	Loops     *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats     estats.Stats     `desc:"contains computed statistic values"`
	Logs      elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Envs      env.Envs         `view:"no-inline" desc:"Environments"`
	Context   axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt  netview.ViewUpdt `view:"inline" desc:"netview update parameters"`
	Decoder   decoder.SoftMax  `desc:"decoder for better output"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
	Comm     *mpi.Comm   `view:"-" desc:"mpi communicator"`
	AllDWts  []float32   `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts  []float32   `view:"-" desc:"buffer of MPI summed dwt weight changes"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for args when calling methods
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Config.Defaults()
	econfig.Config(&ss.Config, "config.toml")
	if ss.Config.Run.MPI {
		ss.MPIInit()
	}
	if mpi.WorldRank() != 0 {
		ss.Config.Log.SaveWts = false
		ss.Config.Log.NetData = false
	}
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.InitRndSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	newEnv := (len(ss.Envs) == 0)

	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn, tst *FWorld
		if newEnv {
			trn = &FWorld{}
			tst = &FWorld{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*FWorld)
			tst = ss.Envs.ByModeDi(etime.Test, di).(*FWorld)
		}

		trn.Defaults()
		trn.Nm = env.ModeDi(etime.Train, di)
		trn.RndSeed = 73 + int64(di)*73
		if ss.Config.Env.Env != nil {
			params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
		}
		trn.Config(ss.Config.Run.NTrials)
		trn.Init(0)
		trn.Validate()

		tst.Defaults()
		tst.Nm = env.ModeDi(etime.Test, di)
		tst.RndSeed = 181 + int64(di)*181
		if ss.Config.Env.Env != nil {
			params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
		}
		tst.Config(ss.Config.Run.NTrials)
		tst.Init(0)
		tst.Validate()

		ss.Envs.Add(trn, tst)
		if di == 0 {
			ss.ConfigPVLV(trn)
		}
	}
}

func (ss *Sim) ConfigPVLV(trn *FWorld) {
	pv := &ss.Context.PVLV
	pv.Drive.NActive = uint32(trn.NDrives) + 1
	pv.Drive.DriveMin = 0.5 // 0.5 -- should be
	pv.Drive.Base.SetAll(1)
	pv.Drive.Base.Set(0, 0.5) // curiosity
	pv.Drive.Tau.SetAll(100)
	pv.Drive.Tau.Set(0, 0)
	pv.Drive.USDec.SetAll(0.5)
	pv.Drive.USDec.Set(0, 0)
	pv.Drive.Update()
	pv.Effort.Gain = 0.01
	pv.Effort.Max = 40
	pv.Effort.MaxNovel = 40
	pv.Effort.MaxPostDip = 8
	pv.Urgency.U50 = 40
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*FWorld)

	net.InitName(net, "Emery")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	full := prjn.NewFull()
	one2one := prjn.NewOneToOne()
	sameu := prjn.NewPoolSameUnit()
	sameu.SelfCon = false
	p1to1 := prjn.NewPoolOneToOne()
	_ = p1to1

	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1
	_ = rndcut

	// prjnClass := "PFCPrjn"

	var ff, fb emer.Prjn

	nPerAng := 5 // 5*1 impairs M1P prediction somewhat -- nothing else
	nPerDepth := 1
	rfDepth := 6
	rfWidth := 3
	stdHidX := 6 // smaller 10 -> 6 impairs M1P prediction somewhat, not actmatch etc
	stdHidY := 6

	nUSs := ev.NDrives + 1 // first US / drive is novelty / curiosity
	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	popY := 4
	popX := 4
	space := float32(2)
	ny := ev.NYReps

	rect := prjn.NewRect()
	rect.Size.Set(rfWidth, rfDepth) // 6 > 8 > smaller
	rect.Scale.Set(1.0/float32(nPerAng), 1.0/float32(nPerDepth))
	_ = rect

	fsz := 1 + 2*ev.FoveaSize
	// popsize = 12

	////////////////////////////////////////
	// input / output layers:

	vSgpi, effort, effortP, urgency, pvPos, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcUS, ofcUSCT, ofcUSPTp, ofcVal, ofcValCT, ofcValPTp, accCost, accCostCT, accCostPTp, accUtil, sc, notMaint := net.AddBOA(&ss.Context, nUSs, ny, popY, popX, nuBgY, nuBgX, nuCtxY, nuCtxX, space)
	_, _ = accUtil, urgency

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

	// 2D MSTd -- much simpler:
	mstdSz := evec.Vec2i{X: (ev.NFOVRays - (rfWidth - 1)) * nPerAng, Y: (ev.DepthSize - (rfDepth - 1)) * nPerDepth}
	mstd, mstdCT := net.AddSuperCT2D("MSTd", "MSTd", mstdSz.Y, mstdSz.X, space, one2one) // def one2one
	mstdCT.SetClass("MSTd CTCopy")
	// mstdP.SetClass("MSTd")
	// net.ConnectCTSelf(mstdCT, full) // not needed or beneficial for simple 1 step move pred
	net.ConnectToPulv(mstd, mstdCT, v2wdP, full, rect, "")
	net.ConnectLayers(v2wd, mstd, rect, axon.ForwardPrjn)

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
	pccP := net.AddPulvForSuper(pcc, space)
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

	//////////////////////////////////////
	// Action prediction

	alm, almCT, almPT, almPTp, almMD := net.AddPFC2D("ALM", "MD", nuCtxY, nuCtxX, true, space)
	_ = almPT

	net.ConnectLayers(vSgpi, almMD, full, axon.InhibPrjn)
	// net.ConnectToMatrix(alm, vSmtxGo, full) // todo: explore
	// net.ConnectToMatrix(alm, vSmtxNo, full)

	net.ConnectToPFCBidir(m1, m1P, alm, almCT, almPTp, full) // alm predicts m1

	///////////////////////////////////////////
	// CS -> BLA, OFC

	net.ConnectToSC1to1(v1f, sc)

	net.ConnectCSToBLAPos(v1f, blaPosAcq, blaNov)
	net.ConnectToBLAExt(v1f, blaPosExt, full)

	net.ConnectToBLAAcq(v1f, blaNegAcq, full)
	net.ConnectToBLAExt(v1f, blaNegExt, full)

	// OFCus predicts v1f
	// net.ConnectToPFCBack(v1f, csP, ofcUS, ofcUSCT, ofcUSPTp, full)

	///////////////////////////////////////////
	// OFC, ACC, ALM predicts dist

	// todo: a more dynamic US rep is needed to drive predictions in OFC
	// using distance and effort here in the meantime
	net.ConnectToPFCBack(effort, effortP, ofcUS, ofcUSCT, ofcUSPTp, full)
	net.ConnectToPFCBack(effort, effortP, ofcVal, ofcValCT, ofcValPTp, full)

	// note: effort, urgency for accCost already set in AddBOA
	net.ConnectToPFC(pcc, pccP, accCost, accCostCT, accCostPTp, full)

	//	alm predicts all effort, cost, sensory state vars
	net.ConnectToPFC(effort, effortP, alm, almCT, almPTp, full)
	net.ConnectToPFC(pcc, pccP, alm, almCT, almPTp, full)

	///////////////////////////////////////////
	// ALM, M1 <-> OFC, ACC

	// super contextualization based on action, not good?
	// net.BidirConnectLayers(ofcUS, alm, full)
	// net.BidirConnectLayers(accCost, alm, full)

	// action needs to know if maintaining a goal or not
	// using ofcVal and accCost as representatives
	net.ConnectLayers(ofcValPTp, alm, full, axon.ForwardPrjn).SetClass("ToALM")
	net.ConnectLayers(accCostPTp, alm, full, axon.ForwardPrjn).SetClass("ToALM")
	net.ConnectLayers(notMaint, alm, full, axon.ForwardPrjn).SetClass("ToALM")

	net.ConnectLayers(ofcValPTp, m1, full, axon.ForwardPrjn).SetClass("ToM1")
	net.ConnectLayers(accCostPTp, m1, full, axon.ForwardPrjn).SetClass("ToM1")
	net.ConnectLayers(notMaint, m1, full, axon.ForwardPrjn).SetClass("ToM1")

	// full shortcut -- todo: test removing
	net.ConnectLayers(ofcValPTp, vl, full, axon.ForwardPrjn).SetClass("ToVL")
	net.ConnectLayers(accCostPTp, vl, full, axon.ForwardPrjn).SetClass("ToVL")
	net.ConnectLayers(notMaint, vl, full, axon.ForwardPrjn).SetClass("ToVL")

	////////////////////////////////////////////////
	// position

	v2wd.PlaceRightOf(pvPos, space)

	v1f.PlaceRightOf(v2wdP, space)
	// v2fd.PlaceBehind(v1f, space)
	v2wdP.PlaceRightOf(v2wd, space)

	s1s.PlaceBehind(v1f, space)
	// s1sP.PlaceBehind(s1s, space)

	hd.PlaceBehind(s1s, space)
	hdP.PlaceBehind(hd, space)

	s2v.PlaceBehind(hdP, space)
	s2vCT.PlaceRightOf(s2v, space)

	vl.PlaceRightOf(sc, space)
	act.PlaceBehind(vl, space)

	mstd.PlaceRightOf(v1f, space)

	// cipl.PlaceRightOf(mstd, space)

	pcc.PlaceRightOf(mstd, space)
	sma.PlaceRightOf(pcc, space)
	m1.PlaceRightOf(sma, space)

	alm.PlaceRightOf(accUtil, space)
	notMaint.PlaceRightOf(alm, space)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	ss.Net.InitWts(ctx)
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll() // first hard-coded defaults
	if ss.Config.Params.Network != nil {
		ss.Params.SetNetworkMap(ss.Net, ss.Config.Params.Network)
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if ss.Config.GUI {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
	ss.Loops.ResetCounters()
	ss.InitRndSeed(0)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed(run int) {
	ss.RndSeeds.Set(run)
	ss.RndSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	totND := ss.Config.Run.NData * mpi.WorldSize() // both sources of data parallel
	totTrls := int(mat32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(totND)))
	trls := totTrls / mpi.WorldSize()

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, 200)

	// note: needs a lot of data for good actrfs -- is relatively fast under mpi
	man.AddStack(etime.Test).
		AddTime(etime.Epoch, ss.Config.Run.NTstEpochs).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, 200)

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
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	// note: plusPhase is shared between all stacks!
	plusPhase, _ := man.Stacks[etime.Train].Loops[etime.Cycle].EventByName("PlusPhase")
	plusPhase.OnEvent.InsertBefore("PlusPhase:Start", "TakeAction", func() {
		// note: critical to have this happen *after* MinusPhase:End and *before* PlusPhase:Start
		// because minus phase end has gated info, and plus phase start applies action input
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
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			if ss.Config.Run.MPI {
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
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	// this is actually fairly expensive
	man.GetLoop(etime.Test, etime.Trial).OnEnd.Add("ActRFs", func() {
		ss.UpdateActRFs()
	})
	// man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("CheckEpc", func() {
	// 	if ss.Args.Bool("actrfs") {
	// 		trnEpc := man.Stacks[etime.Test].Loops[etime.Epoch].Counter.Cur
	// 		fmt.Printf("epoch: %d\n", trnEpc)
	// 	}
	// })

	// Save weights to file at end, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
	})

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PctCortex", func() {
		trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		ss.Config.Env.CurPctCortex(trnEpc)
	})

	////////////////////////////////////////////
	// GUI

	if !ss.Config.GUI {
		if ss.Config.Log.NetData {
			man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
				ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
			})
		}
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net)
		axon.LooperUpdtPlots(man, &ss.GUI)
		for _, m := range man.Stacks {
			m.Loops[etime.Cycle].OnEnd.Prepend("GUI:CounterUpdt", func() {
				ss.NetViewCounters(etime.Cycle)
			})
			m.Loops[etime.Trial].OnEnd.Prepend("GUI:CounterUpdt", func() {
				ss.NetViewCounters(etime.Trial)
			})
		}
		man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("UpdateWorldGui", func() {
			ev := ss.Envs.ByModeDi(etime.Train, 0).(*FWorld)
			ev.UpdateWorldGui()
		})
	}

	if ss.Config.Debug {
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
	axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) TakeAction(net *axon.Network) {
	ctx := &ss.Context
	mtxLy := ss.Net.AxonLayerByName("VsMtxGo")
	vlly := ss.Net.AxonLayerByName("VL")
	threshold := float32(0.1)

	for di := 0; di < ss.Config.Run.NData; di++ {
		diu := uint32(di)
		ev := ss.Envs.ByModeDi(ctx.Mode, di).(*FWorld)
		justGated := mtxLy.AnyGated(diu) // not updated until plus phase: ss.Context.PVLV.VSMatrix.JustGated.IsTrue()
		hasGated := axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0
		ss.Stats.SetStringDi("PrevAction", di, ss.Stats.StringDi("ActAction", di))
		csGated := (justGated && !axon.PVLVHasPosUS(ctx, diu))
		deciding := !csGated && !hasGated && (axon.GlbV(ctx, diu, axon.GvACh) > threshold && mtxLy.Pool(0, diu).AvgMax.SpkMax.Cycle.Max > threshold) // give it time
		ss.Stats.SetFloat32Di("Deciding", di, bools.ToFloat32(deciding))
		if csGated || deciding {
			act := "CSGated"
			if !csGated {
				act = "Deciding"
			}
			ss.Stats.SetStringDi("Debug", di, act)
			ev.Action("None", nil)
			ss.ApplyAction(di)
			ss.Stats.SetStringDi("ActAction", di, "None")
			ss.Stats.SetStringDi("Instinct", di, "None")
			ss.Stats.SetStringDi("NetAction", di, act)
			ss.Stats.SetFloatDi("ActMatch", di, 1)                // whatever it is, it is ok
			vlly.Pool(0, uint32(di)).Inhib.Clamped.SetBool(false) // not clamped this trial
		} else {
			ss.Stats.SetStringDi("Debug", di, "acting")
			netAct, anm := ss.DecodeAct(ev, di)
			genAct, urgency := ev.InstinctAct(justGated, hasGated)
			_ = urgency
			genActNm := ev.Acts[genAct]
			ss.Stats.SetStringDi("NetAction", di, anm) // ev.Acts[nact]
			ss.Stats.SetStringDi("Instinct", di, genActNm)
			if netAct == genAct {
				ss.Stats.SetFloatDi("ActMatch", di, 1)
			} else {
				ss.Stats.SetFloatDi("ActMatch", di, 0)
			}

			actAct := genAct
			// if erand.BoolP32(urgency, -1) {
			// 	actAct = ss.Stats.String("Instinct")
			// } else if erand.BoolP(ss.PctCortex, -1) {
			// 	actAct = ss.Stats.String("NetAction")
			// }
			if ss.Stats.FloatDi("CortexDriving", di) > 0 {
				actAct = netAct
			}
			actActNm := ev.Acts[actAct]
			ss.Stats.SetStringDi("ActAction", di, actActNm)

			ev.Action(actActNm, nil)
		}
	}
	ss.Net.ApplyExts(ctx)
	ss.Net.GPU.SyncPoolsToGPU()
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *FWorld, di int) (int, string) {
	vt := ss.Stats.SetLayerTensor(ss.Net, "VL", "CaSpkP", di)
	return ev.DecodeAct(vt)
}

func (ss *Sim) ApplyAction(di int) {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByModeDi(ss.Context.Mode, di).(*FWorld)
	ap := ev.State("Action")
	ly := net.AxonLayerByName("Act")
	ly.ApplyExt(ctx, uint32(di), ap)
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "HeadDir", "Action"}
	lays := []string{"V2Wd", "V2Fd", "V1F", "S1S", "HeadDir", "Act"}
	net.InitExt(ctx)

	for di := uint32(0); di < uint32(ss.Config.Run.NData); di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, int(di)).(*FWorld)
		ev.Step()
		// todo:
		// if ev.Time == 0 {
		// 	ss.Stats.SetFloat32Di("CortexDriving", int(di), bools.ToFloat32(erand.BoolP32(ss.Config.Run.PctCortex, -1)))
		// }
		ss.Stats.SetStringDi("TrialName", int(di), ev.String())
		for i, lnm := range lays {
			ly := ss.Net.AxonLayerByName(lnm)
			if ly == nil {
				continue
			}
			pats := ev.State(states[i])
			if pats != nil {
				ly.ApplyExt(ctx, di, pats)
			}
		}
		ss.ApplyPVLV(ctx, ev, di)
	}
	net.ApplyExts(ctx)
}

// ApplyPVLV applies current PVLV values to Context.PVLV,
// from given trial data.
func (ss *Sim) ApplyPVLV(ctx *axon.Context, ev *FWorld, di uint32) {
	ctx.PVLV.EffortUrgencyUpdt(ctx, di, &ss.Net.Rand, ev.LastEffort)
	ctx.PVLVInitUS(di)
	posUSs := ev.State("PosUSs").(*etensor.Float32)
	for i, us := range posUSs.Values {
		if us > 0 {
			ctx.PVLVSetUS(di, axon.Positive, i, us)
		}
	}
	negUSs := ev.State("NegUSs").(*etensor.Float32)
	for i, us := range negUSs.Values {
		if us > 0 {
			ctx.PVLVSetUS(di, axon.Negative, i, us)
			// ctx.NeuroMod.HasRew.SetBool(true) // todo: counting as full US for now
		}
	}
	axon.PVLVDriveUpdt(ctx, di)
	ctx.PVLVStepStart(di, &ss.Net.Rand)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	for di := 0; di < int(ctx.NetIdxs.NData); di++ {
		ss.Envs.ByModeDi(etime.Train, di).Init(0)
		ss.Envs.ByModeDi(etime.Test, di).Init(0)
		axon.DrivesToBaseline(ctx, uint32(di))
		// axon.Drive.Drives.SetAll(0.5) // start lower
	}
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWts(ctx)
	ss.InitStats()
	ss.StatCounters(0)
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
	ss.Stats.SetInt("Di", 0)
	ss.Stats.SetString("PrevAction", "")
	ss.Stats.SetString("ActAction", "")
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Stats.SetFloat("PctCortex", 0)
	ss.Stats.SetFloat("Dist", 0)
	ss.Stats.SetFloat("Drive", 0)
	ss.Stats.SetFloat("CS", 0)
	ss.Stats.SetFloat("US", 0)
	ss.Stats.SetFloat("HasRew", 0)
	ss.Stats.SetString("NetAction", "")
	ss.Stats.SetString("Instinct", "")
	ss.Stats.SetString("ActAction", "")
	ss.Stats.SetFloat("JustGated", 0)
	ss.Stats.SetFloat("Should", 0)
	ss.Stats.SetFloat("GateUS", 0)
	ss.Stats.SetFloat("GateCS", 0)
	ss.Stats.SetFloat("Deciding", 0)
	ss.Stats.SetFloat("GatedEarly", 0)
	ss.Stats.SetFloat("MaintEarly", 0)
	ss.Stats.SetFloat("GatedAgain", 0)
	ss.Stats.SetFloat("WrongCSGate", 0)
	ss.Stats.SetFloat("AChShould", 0)
	ss.Stats.SetFloat("AChShouldnt", 0)
	ss.Stats.SetFloat("Rew", 0)
	ss.Stats.SetFloat("DA", 0)
	ss.Stats.SetFloat("RewPred", 0)
	ss.Stats.SetFloat("DA_NR", 0)
	ss.Stats.SetFloat("RewPred_NR", 0)
	ss.Stats.SetFloat("DipSum", 0)
	ss.Stats.SetFloat("GiveUp", 0)
	ss.Stats.SetFloat("Urge", 0)
	ss.Stats.SetFloat("ActMatch", 0)
	ss.Stats.SetFloat("AllGood", 0)
	lays := ss.Net.LayersByType(axon.PTMaintLayer)
	for _, lnm := range lays {
		ss.Stats.SetFloat("Maint"+lnm, 0)
		ss.Stats.SetFloat("MaintFail"+lnm, 0)
		ss.Stats.SetFloat("PreAct"+lnm, 0)
	}
	ss.Stats.SetString("Debug", "") // special debug notes per trial
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters(di int) {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.ActionStatsDi(di)
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ss.Stats.SetFloat32("PctCortex", ss.Config.Env.PctCortex)
	ss.Stats.SetString("TrialName", ss.Stats.StringDi("TrialName", di))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.GUI.ViewUpdt.View == nil {
		return
	}
	di := ss.GUI.ViewUpdt.View.Di
	if tm == etime.Trial {
		ss.TrialStats(di) // get trial stats for current di
	}
	ss.StatCounters(di)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Cycle", "PrevAction", "NetAction", "Instinct", "ActAction", "ActMatch"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	diu := uint32(di)
	ctx := &ss.Context
	ss.GatedStats(di)
	ss.MaintStats(di)

	out := ss.Net.AxonLayerByName("VL")
	v2wdP := ss.Net.AxonLayerByName("V2WdP")

	ss.Stats.SetFloat("TrlCorSim", float64(v2wdP.Vals[di].CorSim.Cor))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr(ctx)[di])

	// note: most stats computed in TakeAction
	ss.Stats.SetFloat("TrlErr", 1-ss.Stats.Float("ActMatch"))

	nan := math.NaN()
	if axon.PVLVHasPosUS(ctx, diu) {
		ss.Stats.SetFloat32("DA", axon.GlbV(ctx, diu, axon.GvDA))
		ss.Stats.SetFloat32("RewPred", axon.GlbV(ctx, diu, axon.GvRewPred)) // gets from VSPatch or RWPred etc
		ss.Stats.SetFloat("DA_NR", nan)
		ss.Stats.SetFloat("RewPred_NR", nan)
		ss.Stats.SetFloat32("Rew", axon.GlbV(ctx, diu, axon.GvRew))
	} else {
		ss.Stats.SetFloat32("DA_NR", axon.GlbV(ctx, diu, axon.GvDA))
		ss.Stats.SetFloat32("RewPred_NR", axon.GlbV(ctx, diu, axon.GvRewPred))
		ss.Stats.SetFloat("DA", nan)
		ss.Stats.SetFloat("RewPred", nan)
		ss.Stats.SetFloat("Rew", nan)
	}

	ss.Stats.SetFloat32("DipSum", axon.GlbV(ctx, diu, axon.GvLHbDipSum))
	ss.Stats.SetFloat32("GiveUp", axon.GlbV(ctx, diu, axon.GvLHbGiveUp))
	ss.Stats.SetFloat32("Urge", axon.GlbV(ctx, diu, axon.GvUrgency))
	ss.Stats.SetFloat32("ACh", axon.GlbV(ctx, diu, axon.GvACh))
	ss.Stats.SetFloat32("AChRaw", axon.GlbV(ctx, diu, axon.GvAChRaw))

	// epc := env.Epoch.Cur
	// if epc > ss.ConfusionEpc {
	// 	ss.Stats.Confusion.Incr(ss.Stats.Int("TrlCatIdx"), ss.Stats.Int("TrlRespIdx"))
	// }
}

// ActionStatsDi copies the action info from given data parallel index
// into the global action stats
func (ss *Sim) ActionStatsDi(di int) {
	if _, has := ss.Stats.Strings[estats.DiName("NetAction", di)]; !has {
		return
	}
	ss.Stats.SetString("NetAction", ss.Stats.StringDi("NetAction", di))
	ss.Stats.SetString("Instinct", ss.Stats.StringDi("Instinct", di))
	ss.Stats.SetFloat("ActMatch", ss.Stats.FloatDi("ActMatch", di))
	ss.Stats.SetString("ActAction", ss.Stats.StringDi("ActAction", di))
}

// GatedStats updates the gated states
func (ss *Sim) GatedStats(di int) {
	ctx := &ss.Context
	diu := uint32(di)
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*FWorld)
	justGated := axon.GlbV(ctx, diu, axon.GvVSMatrixJustGated) > 0
	hasGated := axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0
	nan := mat32.NaN()
	ss.Stats.SetString("Debug", ss.Stats.StringDi("Debug", di))
	ss.ActionStatsDi(di)

	ss.Stats.SetFloat32("JustGated", bools.ToFloat32(justGated))
	ss.Stats.SetFloat32("Should", bools.ToFloat32(ev.ShouldGate))
	ss.Stats.SetFloat32("HasGated", bools.ToFloat32(hasGated))
	ss.Stats.SetFloat32("GateUS", nan)
	ss.Stats.SetFloat32("GateCS", nan)
	ss.Stats.SetFloat32("GatedEarly", nan)
	ss.Stats.SetFloat32("MaintEarly", nan)
	ss.Stats.SetFloat32("GatedAgain", nan)
	ss.Stats.SetFloat32("WrongCSGate", nan)
	ss.Stats.SetFloat32("AChShould", nan)
	ss.Stats.SetFloat32("AChShouldnt", nan)
	hasPos := axon.PVLVHasPosUS(ctx, diu)
	// if justGated {
	// 	ss.Stats.SetFloat32("WrongCSGate", bools.ToFloat32(!ev.PosHasDriveUS()))
	// }
	if ev.ShouldGate {
		if hasPos {
			ss.Stats.SetFloat32("GateUS", bools.ToFloat32(justGated))
		} else {
			ss.Stats.SetFloat32("GateCS", bools.ToFloat32(justGated))
		}
	} else {
		if hasGated {
			ss.Stats.SetFloat32("GatedAgain", bools.ToFloat32(justGated))
		} else { // !should gate means early..
			ss.Stats.SetFloat32("GatedEarly", bools.ToFloat32(justGated))
		}
	}
	// todo
	// We get get ACh when new CS or Rew
	// if hasPos || ev.LastCS != ev.CS {
	// 	ss.Stats.SetFloat32("AChShould", axon.GlbV(ctx, diu, axon.GvACh))
	// } else {
	// 	ss.Stats.SetFloat32("AChShouldnt", axon.GlbV(ctx, diu, axon.GvACh))
	// }
}

// MaintStats updates the PFC maint stats
func (ss *Sim) MaintStats(di int) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*FWorld)
	// should be maintaining while going forward
	isFwd := ev.LastAct == ev.ActMap["Forward"]
	isCons := ev.LastAct == ev.ActMap["Consume"]
	actThr := float32(0.05) // 0.1 too high
	net := ss.Net
	// hasMaint := false
	lays := net.LayersByType(axon.PTMaintLayer)
	// hasMaint := false
	for _, lnm := range lays {
		mnm := "Maint" + lnm
		fnm := "MaintFail" + lnm
		pnm := "PreAct" + lnm
		ptly := net.AxonLayerByName(lnm)
		var mact float32
		if ptly.Is4D() {
			for pi := uint32(1); pi < ptly.NPools; pi++ {
				avg := ptly.Pool(pi, uint32(di)).AvgMax.Act.Plus.Avg
				if avg > mact {
					mact = avg
				}
			}
		} else {
			mact = ptly.Pool(0, uint32(di)).AvgMax.Act.Plus.Avg
		}
		overThr := mact > actThr
		// if overThr {
		// 	hasMaint = true
		// }
		ss.Stats.SetFloat32(pnm, mat32.NaN())
		ss.Stats.SetFloat32(mnm, mat32.NaN())
		ss.Stats.SetFloat32(fnm, mat32.NaN())
		if isFwd {
			ss.Stats.SetFloat32(mnm, mact)
			ss.Stats.SetFloat32(fnm, bools.ToFloat32(!overThr))
		} else if !isCons {
			ss.Stats.SetFloat32(pnm, bools.ToFloat32(overThr))
		}
	}
	// if hasMaint { // todo
	// 	ss.Stats.SetFloat32("MaintEarly", bools.ToFloat32(!ev.PosHasDriveUS()))
	// }
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "PctCortex")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.Trial, "Drive", "CS", "Dist", "US", "HasRew")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "NetAction", "Instinct", "ActAction")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	// Copy over Testing items
	// ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr")

	axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigActRFs()

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer, axon.CeMLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)

	axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net, etime.Test, etime.Cycle)
	// ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")
	// ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.AllModes, etime.Cycle, "TargetLayer")

	ss.Logs.PlotItems("PctCortex", "ActMatch", "Rew", "GateUS", "GateCS", "V2WdP_CorSim")

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
	ss.Logs.AddStatAggItem("AllGood", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ActMatch", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("JustGated", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Should", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GateUS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GateCS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Deciding", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GatedEarly", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("MaintEarly", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GatedAgain", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("WrongCSGate", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AChShould", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AChShouldnt", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GiveUp", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("DipSum", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Urge", etime.Run, etime.Epoch, etime.Trial)

	// Add a special debug message -- use of etime.Debug triggers
	// inclusion
	ss.Logs.AddStatStringItem(etime.Debug, etime.Trial, "Debug")

	lays := ss.Net.LayersByType(axon.PTMaintLayer)
	for _, lnm := range lays {
		nm := "Maint" + lnm
		ss.Logs.AddStatAggItem(nm, etime.Run, etime.Epoch, etime.Trial)
		nm = "MaintFail" + lnm
		ss.Logs.AddStatAggItem(nm, etime.Run, etime.Epoch, etime.Trial)
		nm = "PreAct" + lnm
		ss.Logs.AddStatAggItem(nm, etime.Run, etime.Epoch, etime.Trial)
	}
	li := ss.Logs.AddStatAggItem("Rew", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("ACh", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("AChRaw", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("RewPred", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA_NR", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("RewPred_NR", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false

	ev := ss.Envs.ByModeDi(etime.Train, 0).(*FWorld)
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
			Plot:  false,
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
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}

	if ss.Config.Run.MPI && time == etime.Epoch { // gather data for trial level at epoch
		ss.Logs.MPIGatherTableRows(mode, etime.Trial, ss.Comm)
	}

	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		if mode == etime.Train {
			for di := 0; di < int(ctx.NetIdxs.NData); di++ {
				diu := uint32(di)
				ss.TrialStats(di)
				ss.StatCounters(di)
				ss.Logs.LogRowDi(mode, time, row, di)
				if !axon.PVLVHasPosUS(ctx, diu) && axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0 { // maint
					axon.LayerActsLog(ss.Net, &ss.Logs, di, &ss.GUI)
				}
				if ss.ViewUpdt.View != nil && di == ss.ViewUpdt.View.Di {
					drow := ss.Logs.Table(etime.Debug, time).Rows
					ss.Logs.LogRow(etime.Debug, time, drow)
					if ss.StopOnSeq {
						hasRew := axon.GlbV(ctx, uint32(di), axon.GvHasRew) > 0
						if hasRew {
							ss.Loops.Stop(etime.Trial)
						}
					}
					ss.GUI.UpdateTableView(etime.Debug, etime.Trial)
				}
				// if ss.Stats.Float("GatedEarly") > 0 {
				// 	fmt.Printf("STOPPED due to gated early: %d  %g\n", ev.US, ev.Rew)
				// 	ss.Loops.Stop(etime.Trial)
				// }
				// ev := ss.Envs.ByModeDi(etime.Train, di).(*Approach)
				// if ss.StopOnErr && trnEpc > 5 && ss.Stats.Float("MaintEarly") > 0 {
				// 	fmt.Printf("STOPPED due to early maint for US: %d\n", ev.US)
				// 	ss.Loops.Stop(etime.Trial)
				// }
			}
			return // don't do reg
		}
		// case mode == etime.Train && time == etime.Epoch:
		// 	axon.LayerActsLogAvg(ss.Net, &ss.Logs, &ss.GUI, true) // reset recs
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
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*FWorld)
	for _, trg := range ss.Config.Log.RFTargs {
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
		for _, trg := range ss.Config.Log.RFTargs {
			arfs = append(arfs, lnm+":"+trg)
		}
	}
	ss.Stats.InitActRFs(ss.Net, arfs, "ActM")
}

// UpdateActRFs updates position activation rf's
func (ss *Sim) UpdateActRFs() {
	ctx := &ss.Context
	for di := 0; di < int(ctx.NetIdxs.NData); di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, di).(*FWorld)
		for _, trg := range ss.Config.Log.RFTargs {
			mt := ss.Stats.F32TensorDi(trg, di)
			mt.SetZeros()
			switch trg {
			case "Pos":
				mt.Set([]int{ev.PosI.Y, ev.PosI.X}, 1)
			case "Act":
				mt.Set1D(ev.LastAct, 1)
			case "HdDir":
				mt.Set1D(ev.HeadDir/ev.MotAngInc, 1)
				// case "Rot":
				// 	mt.Set1D(1+ev.RotAng/ev.NMotAngles, 1)
			}
		}
		ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01, di)
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
	for _, trg := range ss.Config.Log.RFTargs {
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
	for _, trg := range ss.Config.Log.RFTargs {
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
	cam.Pose.Pos.Set(0, 1.5, 2.5)
	cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Emery"
	ss.GUI.MakeWindow(ss, "eboa", title, `Full brain predictive learning in navigational / survival environment. See <a href="https://github.com/ccnlab/map-nav/blob/master/sims/eboa/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.02
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
			gi.OpenURL("https://github.com/map-nav/blob/main/sims/eboa/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win
}

func (ss *Sim) RunGUI() {
	ss.Init()
	win := ss.ConfigGui()
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*FWorld)
	fwin := ev.ConfigWorldGui()
	fwin.GoStartEventLoop()
	win.StartEventLoop()
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
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWts {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name()

	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	tmr := timer.Time{}
	tmr.Start()

	ss.Loops.Run(etime.Train)

	tmr.Stop()
	fmt.Printf("Total Time: %6.3g\n", tmr.TotalSecs())
	ss.Net.TimerReport()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
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
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.Config.Run.MPI {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
// includes all other long adapting factors too: DTrgAvg, ActAvg, etc
func (ss *Sim) CollectDWts(net *axon.Network) {
	net.CollectDWts(&ss.Context, &ss.AllDWts)
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	ctx := &ss.Context
	if ss.Config.Run.MPI {
		ss.CollectDWts(ss.Net)
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ctx, ss.SumDWts, mpi.WorldSize())
	}
	ss.Net.WtFmDWt(ctx)
}
