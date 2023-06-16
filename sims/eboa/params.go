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
					"Layer.Acts.Clamp.Ge": "1.5", // .6 was
				}},
			{Sel: ".SuperLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Inhib.Layer.On":       "true",
					"Layer.Inhib.Layer.Gi":       "1.0",
					"Layer.Inhib.Pool.Gi":        "1.0",
				}},
			{Sel: ".CTLayer", Desc: "corticothalamic context -- using markovian copy params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.12",
					"Layer.CT.GeGain":            "1.0",
					"Layer.CT.DecayTau":          "0",
					"Layer.Inhib.Layer.Gi":       "2.4",
					"Layer.Inhib.Pool.Gi":        "2.4",
					"Layer.Acts.Decay.Act":       "0.0",
					"Layer.Acts.Decay.Glong":     "0.0",
					//					"Layer.Acts.Sahp.Gbar":     "0.1",
				}},
			{Sel: ".CTCopy", Desc: "single-step copy params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.12",
					"Layer.CT.GeGain":            "1.0",
					"Layer.CT.DecayTau":          "0",
					"Layer.Inhib.Layer.Gi":       "2.0",
					"Layer.Acts.NMDA.Gbar":       "0.006", // std
					"Layer.Acts.NMDA.Tau":        "100",   // std
				}},
			{Sel: ".CTInteg", Desc: "time integration params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.12",
					"Layer.CT.GeGain":            "1.0",
					"Layer.CT.DecayTau":          "50",
					"Layer.Inhib.Layer.Gi":       "2.8",
					"Layer.Acts.MaintNMDA.Gbar":  "0.007", // 0.007 best, but 0.01 > lower if reg nmda weak
					"Layer.Acts.MaintNMDA.Tau":   "200",   // 200 > 100 > 300
					"Layer.Acts.NMDA.Gbar":       "0.007", // 0.007 matching maint best
					"Layer.Acts.NMDA.Tau":        "200",   // 200 > 100
				}},
			{Sel: ".PulvinarLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "0.8",  // was 0.9 > 1.0
					"Layer.Pulv.DriveScale":         "0.1",  // 0.1 basically same as 0.15
					"Layer.Acts.Decay.Act":          "0.0",  // clear
					"Layer.Acts.Decay.Glong":        "0.0",  //
					"Layer.Acts.Decay.AHP":          "0.0",  //
					"Layer.Learn.RLRate.On":         "true", // beneficial for trace
					"Layer.Learn.RLRate.SigmoidMin": "1",
				}},
			{Sel: ".Depth", Desc: "depth layers not using pool",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1", // 0.1 eventually
					"Layer.Inhib.Layer.Gi":       "1.0",
				}},
			{Sel: "#V2WdP", Desc: "weaker inhibition for pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.8",
				}},
			{Sel: ".Fovea", Desc: "fovea has both",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.15",
					"Layer.Inhib.Layer.On":       "true", // layer too
					"Layer.Inhib.Layer.Gi":       "1",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.Gi":        "1",
				}},
			{Sel: ".S1S", Desc: "lower inhib, higher act",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "0.9", // some weaker global inhib
					"Layer.Inhib.ActAvg.Nominal": "0.3",
				}},
			{Sel: ".HeadDir", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.13", // 0.13 actual
					"Layer.Inhib.Layer.Gi":       "0.9",
				}},
			{Sel: ".MSTd", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Inhib.Layer.Gi":       "1.2",
				}},
			{Sel: "#MSTdCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "2.8", // 2.0 is fine vs. higher, > lower
					"Layer.CT.GeGain":            "1",
					"Layer.Inhib.ActAvg.Nominal": "0.05",
				}},
			{Sel: ".PCC", Desc: "PCC general",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
				}},
			{Sel: "#PCC", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.06",
					"Layer.Inhib.Layer.Gi":       "0.9", // was 1.0
				}},
			{Sel: "#PCCCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Inhib.Layer.Gi":       "2.8",
				}},
			{Sel: ".S2V", Desc: "S2V general",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1", // CT = 0.1 at end
				}},
			{Sel: "#S2V", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.08", // 0.08 actual
				}},
			{Sel: ".SMA", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
				}},
			{Sel: "#SMA", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.08",
					"Layer.Inhib.Layer.Gi":       "0.9",   // was 1.0
					"Layer.Acts.Noise.On":        "false", // turn on for more explore
					"Layer.Acts.Noise.Ge":        "0.001", // actually .001 best
					"Layer.Acts.Noise.Gi":        "0.001",
				}},
			{Sel: "#SMACT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Inhib.Layer.Gi":       "2.0", // 2.0 matches prior act
					"Layer.Acts.Noise.On":        "false",
					"Layer.Acts.Noise.Ge":        "0.001",
					"Layer.Acts.Noise.Gi":        "0.001",
				}},
			{Sel: ".M1", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Inhib.Layer.Gi":       "1.0",
				}},
			{Sel: "#M1P", Desc: "m1 pulvinar",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.06",
				}},
			{Sel: ".Action", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.18", // .18 actual
				}},
			{Sel: "#VL", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.0", // 1.2 = stronger to compete
				}},

			//////////////////////////////////////////////////////////
			// Prjns

			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.005",  // 0.005 ok
					"Prjn.SWts.Adapt.LRate": "0.0001", // 0.001 > 0.01 > 0.0001
					"Prjn.SWts.Init.SPct":   "1.0",    // .5 ok here, 1 best for larger nets: objrec, lvis
					"Prjn.Learn.Trace.Tau":  "2",      // 2 > 1 for many Pulv CorSims
				}},
			{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".StrongFmSMA", Desc: "stronger from SMA",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.5 > 1.0 > 1.5
				}},
			{Sel: ".CTCtxtPrjn", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.002", // 0.002 > .005 > .001
					"Prjn.Learn.Trace.Tau":  "2",     // late in learning 2 does best
				}},
			{Sel: ".CTFmSuper", Desc: "CT from main super",
				Params: params.Params{
					"Prjn.PrjnScale.Rel":  "1",
					"Prjn.Learn.Learn":    "false",
					"Prjn.SWts.Init.SPct": "0",
					"Prjn.SWts.Init.Mean": "0.8",
					"Prjn.SWts.Init.Var":  "0.0",
				}},
			{Sel: ".FixedCTFmSuper", Desc: "non-learning CT from main super -- for CT time integ",
				Params: params.Params{
					"Prjn.PrjnScale.Rel":  "1",
					"Prjn.Learn.Learn":    "false",
					"Prjn.SWts.Init.SPct": "0",
					"Prjn.SWts.Init.Mean": "0.8",
					"Prjn.SWts.Init.Var":  "0.0",
				}},
			{Sel: ".StrongFF", Desc: "extra strong feedforward activation",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: ".FmPulv", Desc: "default for pulvinar",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".CTToPulv", Desc: "basic main CT to pulivnar",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0",
					"Prjn.PrjnScale.Rel": "1",
				}},
			{Sel: ".CTSelfCtxt", Desc: "CT self context",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5",
				}},
			{Sel: ".CTSelfMaint", Desc: "CT self maintenance",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: "#MSTdToPCC", Desc: "extra strong feedforward activation",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "4.0",
				}},
			// {Sel: ".V1SC", Desc: "v1 shortcut",
			// 	Params: params.Params{
			// 		"Prjn.Learn.LRate.Base": "0.001", //
			// 		"Prjn.PrjnScale.Rel":    "0.5",   // .5 lvis
			// 		"Prjn.SWts.Adapt.On":     "false", // seems better
			// 	}},
			// {Sel: "#S1VToS2V", Desc: "needs more",
			// 	Params: params.Params{
			// 		"Prjn.PrjnScale.Abs": "1.5",
			// 	}},
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
