// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "using default 1 inhib for hidden layers",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.06",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.FFEx0":  "0.15",
					// "Layer.Inhib.Pool.FFEx":       "0.02", // .05 for lvis
					"Layer.Inhib.Layer.FFEx0": "0.15",
					// "Layer.Inhib.Layer.FFEx":      "0.02", //
					"Layer.Act.Gbar.L":      "0.2",
					"Layer.Act.Decay.Act":   "0.0", // todo: explore
					"Layer.Act.Decay.Glong": "0.0",
					"Layer.Act.Clamp.Ge":    "1.0", // .6 was
					"Layer.Act.AK.Gbar":     "1.0", // 0.1 def -- unclear if diff
					"Layer.Act.Mahp.Gbar":   "0.04",
					"Layer.Act.Sahp.Gbar":   "1.0",
				}},
			{Sel: ".Hidden", Desc: "noise? sub-pools",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":    "0.06",
					"Layer.Inhib.ActAvg.AdaptGi": "false", // no!
					"Layer.Inhib.Layer.Gi":       "1.1",
					"Layer.Inhib.Pool.Gi":        "1.1",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Layer.On":       "true", // full layer
				}},
			{Sel: ".CT", Desc: "corticothalamic context",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.06",
					"Layer.CT.GeGain":         "0.5",
					"Layer.CT.DecayTau":       "50",
					"Layer.Inhib.Layer.Gi":    "1.4",
					"Layer.Inhib.Pool.Gi":     "1.4",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Act.KNa.On":        "true",
					"Layer.Act.NMDA.Gbar":     "0.35",
					"Layer.Act.NMDA.Tau":      "300",
					"Layer.Act.GABAB.Gbar":    "0.4",
					"Layer.Act.Decay.Act":     "0.0",
					"Layer.Act.Decay.Glong":   "0.0",
				}},
			{Sel: "TRCLayer", Desc: "",
				Params: params.Params{
					"Layer.TRC.DriveScale":          "0.15", // .15 > .05 default
					"Layer.Act.Decay.Act":           "0.5",
					"Layer.Act.Decay.Glong":         "1", // clear long
					"Layer.Inhib.Pool.FFEx":         "0.0",
					"Layer.Inhib.Layer.FFEx":        "0.0",
					"Layer.Learn.RLrate.On":         "true", // beneficial for trace
					"Layer.Learn.RLrate.SigmoidMin": "1",
				}},
			{Sel: ".Depth", Desc: "depth layers use pool inhibition only",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.08",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "false",
					"Layer.Inhib.Layer.Gi":    "0.8",
					"Layer.Inhib.Pool.Gi":     "0.8",
					"Layer.Inhib.Pool.FFEx":   "0.0",
					"Layer.Inhib.Layer.FFEx":  "0.0",
				}},
			{Sel: ".Fovea", Desc: "fovea has both",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.On":    "true", // layer too
					"Layer.Inhib.Layer.Gi":    "1",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1",
					"Layer.Inhib.Pool.FFEx":   "0.0",
					"Layer.Inhib.Layer.FFEx":  "0.0",
				}},
			{Sel: ".S1S", Desc: "lower inhib, higher act",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1", // some weaker global inhib
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "0.8", // weaker
					"Layer.Inhib.ActAvg.Init": "0.12",
				}},
			{Sel: ".S1V", Desc: "lower inhib, higher act",
				Params: params.Params{
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.On":     "false",
					"Layer.Inhib.ActAvg.Init": "0.05",
				}},
			{Sel: ".Ins", Desc: "pools",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1.1",
					"Layer.Inhib.ActAvg.Init": "0.1",
				}},
			{Sel: ".M1", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Layer.Gi":    "1.1",
				}},
			{Sel: ".MSTd", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.ActAvg.Init": "0.015",
					"Layer.Inhib.Layer.Gi":    "1.1", // 1.1 > 1.0
					"Layer.Inhib.Pool.Gi":     "1.1",
					// "Layer.Inhib.Pool.FFEx":   "0.02", //
					// "Layer.Inhib.Layer.FFEx":  "0.02",
				}},
			{Sel: "#MSTdCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.4",
					"Layer.Inhib.Pool.Gi":     "1.4",
					"Layer.Inhib.ActAvg.Init": "0.08",
				}},
			{Sel: "#MSTdP", Desc: "weaker inhibition for pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9", // 0.8 > 0.9
					"Layer.Inhib.Pool.Gi":  "0.9",
				}},
			{Sel: ".cIPL", Desc: "cIPL general",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.Gi":     "1.1",
				}},
			{Sel: "#cIPLCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.4",
					"Layer.Inhib.Pool.Gi":     "1.4",
					"Layer.Inhib.ActAvg.Init": "0.10",
				}},
			{Sel: "#cIPLP", Desc: "weaker inhibition for pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1", // was 0.9
					"Layer.Inhib.Pool.Gi":  "1.1",
				}},
			{Sel: ".PCC", Desc: "PCC general",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.01",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Layer.Gi":    "1.1",
				}},
			{Sel: "#PCCCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.4",
					"Layer.Inhib.Pool.Gi":     "1.4",
					"Layer.Inhib.ActAvg.Init": "0.11",
				}},
			{Sel: "#V2WdP", Desc: "weaker inhibition for pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1", // was 0.8
					"Layer.Inhib.Pool.Gi":  "1.1", // not used
				}},
			{Sel: ".SMA", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Pool.On":     "false",
				}},
			{Sel: "#SMA", Desc: "",
				Params: params.Params{
					"Layer.Act.Noise.On": "true",
					"Layer.Act.Noise.Ge": "0.001",
					"Layer.Act.Noise.Gi": "0.001",
				}},
			{Sel: "#SMACT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Act.Noise.On":      "true",
					"Layer.Act.Noise.Ge":      "0.001",
					"Layer.Act.Noise.Gi":      "0.001",
				}},
			{Sel: "#SMAP", Desc: "pulv",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.Gi":     "1.1",
					"Layer.Inhib.Pool.On":     "true", // independent pathways
					"Layer.Inhib.ActAvg.Init": "0.1",
				}},
			{Sel: "#Act", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
				}},
			{Sel: "#VL", Desc: "VL regular inhib",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Layer.Gi":    "0.8",
					"Layer.Inhib.Pool.FFEx":   "0.0",
					"Layer.Inhib.Layer.FFEx":  "0.0",
				}},
			{Sel: "#M1", Desc: "noise!?",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Layer.Gi":    "1.1", // reg
				}},
			{Sel: "#M1P", Desc: "m1 pulvinar",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Layer.Gi":    "1.0", // weaker pulv
				}},
			{Sel: ".IT", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Pool.On":     "false",
					"Layer.Inhib.Layer.Gi":    "1.1",
				}},
			{Sel: "#ITCT", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "true",
				}},
			{Sel: ".LIP", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1.1",
				}},
			{Sel: "#LIPCT", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
				}},

			//////////////////////////////////////////////////////////
			// Prjns

			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":   "0.04",
					"Prjn.SWt.Adapt.Lrate":    "0.001", // 0.001 > 0.01 > 0.0001
					"Prjn.SWt.Adapt.DreamVar": "0.0",   // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":      "1.0",   // .5 ok here, 1 best for larger nets: objrec, lvis
					// "Prjn.Learn.KinaseCa.UpdtThr": "0.05",  // 0.05 -- was LrnThr
					"Prjn.Learn.Trace.Tau": "2", // 2 > 1 for many TRC CorSims
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".CTBack", Desc: "deep top-down -- stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.2 > 0.5
				}},
			{Sel: ".ActToCT", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: ".Inhib", Desc: "inhibitory projection",
				Params: params.Params{
					"Prjn.Learn.Learn":         "true",  // learned decorrel is good
					"Prjn.Learn.Lrate.Base":    "0.001", // .0001 > .001 -- slower better!
					"Prjn.Learn.Trace.SubMean": "1",     // 1 is *essential* here!
					"Prjn.SWt.Init.Var":        "0.0",
					"Prjn.SWt.Init.Mean":       "0.1",
					"Prjn.SWt.Init.Sym":        "false",
					"Prjn.SWt.Adapt.On":        "false",
					"Prjn.PrjnScale.Abs":       "0.3", // .1 = .2, slower blowup
					"Prjn.IncGain":             "1",   // .5 def
				}},
			{Sel: ".Lateral", Desc: "default for lateral -- not using",
				Params: params.Params{
					"Prjn.SWt.Init.Sym":  "false",
					"Prjn.SWt.Init.Var":  "0",
					"Prjn.PrjnScale.Rel": "0.02", // .02 > .05 == .01 > .1  -- very minor diffs on TE cat
				}},
			{Sel: ".CTCtxt", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.04",
					"Prjn.Trace":            "false",
				}},
			{Sel: ".CTFmSuper", Desc: "CT from main super",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1",
				}},
			{Sel: ".SuperFwd", Desc: "standard superficial forward prjns -- not to output",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
				}},
			{Sel: ".FmPulv", Desc: "default for pulvinar",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".CTSelf", Desc: "CT to CT",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5",
				}},
			{Sel: ".CTToPulv", Desc: "basic main CT to pulivnar -- needs to be stronger -- cons are weak somehow",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0",
					"Prjn.PrjnScale.Rel": "1",
				}},
			{Sel: ".CTToPulv3", Desc: "even stronger abs",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.PrjnScale.Rel": "1",
				}},
			{Sel: ".ToPulv1", Desc: "weaker higher-level pulvinar prjn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".ToPulv2", Desc: "weaker higher-level pulvinar prjn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: ".FwdToPulv", Desc: "feedforward to pulvinar directly",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".CTSelfCtxt", Desc: "CT self context",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5",
				}},
			{Sel: ".CTSelfMaint", Desc: "CT self maintenance",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: "#MSTdTocIPL", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: "#PCCToSMA", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: "#ITToITCT", Desc: "IT likes stronger FmSuper",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
				}},
			{Sel: "#LIPToLIPCT", Desc: "LIP likes stronger FmSuper",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
				}},
			{Sel: "#LIPCTToLIPCT", Desc: "LIP likes stronger CTSelf",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
				}},
			{Sel: ".V1SC", Desc: "v1 shortcut",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.001", //
					"Prjn.PrjnScale.Rel":    "0.5",   // .5 lvis
					"Prjn.SWt.Adapt.On":     "false", // seems better
				}},
		},
	}},
}

// Prjns holds all the special projections
type Prjns struct {
	Prjn4x4Skp2      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x4Skp2Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn4x3Skp2      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x3Skp2Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn3x3Skp1      *prjn.PoolTile `view:"no-inline" desc:"feedforward 3x3 skip 1 topo prjn"`
	Prjn4x4Skp4      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn"`
	Prjn4x4Skp4Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn, recip"`
}

func (pj *Prjns) New() {
	pj.Prjn4x4Skp2 = prjn.NewPoolTile()
	pj.Prjn4x4Skp2.Size.Set(4, 4)
	pj.Prjn4x4Skp2.Skip.Set(2, 2)
	pj.Prjn4x4Skp2.Start.Set(-1, -1)
	pj.Prjn4x4Skp2.TopoRange.Min = 0.5

	pj.Prjn4x4Skp2Recip = prjn.NewPoolTileRecip(pj.Prjn4x4Skp2)

	pj.Prjn4x3Skp2 = prjn.NewPoolTile()
	pj.Prjn4x3Skp2.Size.Set(3, 4)
	pj.Prjn4x3Skp2.Skip.Set(0, 2)
	pj.Prjn4x3Skp2.Start.Set(0, -1)
	pj.Prjn4x3Skp2.TopoRange.Min = 0.5

	pj.Prjn4x3Skp2Recip = prjn.NewPoolTileRecip(pj.Prjn4x3Skp2)

	pj.Prjn3x3Skp1 = prjn.NewPoolTile()
	pj.Prjn3x3Skp1.Size.Set(3, 1)
	pj.Prjn3x3Skp1.Skip.Set(1, 1)
	pj.Prjn3x3Skp1.Start.Set(-1, -1)

	pj.Prjn4x4Skp4 = prjn.NewPoolTile()
	pj.Prjn4x4Skp4.Size.Set(4, 1)
	pj.Prjn4x4Skp4.Skip.Set(4, 1)
	pj.Prjn4x4Skp4.Start.Set(0, 0)
	pj.Prjn4x4Skp4Recip = prjn.NewPoolTileRecip(pj.Prjn4x4Skp4)
}
