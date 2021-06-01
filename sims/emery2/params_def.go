// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "using default 1 inhib for hidden layers",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":            "0.05",
					"Layer.Inhib.ActAvg.Targ":            "0.05",
					"Layer.Inhib.Layer.Gi":               "1.1",
					"Layer.Act.Gbar.L":                   "0.2",
					"Layer.Act.Decay.Act":                "0.0", // both 0 better
					"Layer.Act.Decay.Glong":              "0.0",
					"Layer.Act.Clamp.Rate":               "120",  // 120 == 100 > 150
					"Layer.Act.Dt.TrlAvgTau":             "20",   // 20 > higher for objrec, lvis
					"Layer.Learn.TrgAvgAct.ErrLrate":     "0.02", // 0.02 > 0.05 objrec
					"Layer.Learn.TrgAvgAct.SynScaleRate": "0.01", // 0.01 > 0.005 best for objrec -- needs faster
					"Layer.Learn.TrgAvgAct.TrgRange.Min": "0.5",  // .5 best for Lvis, .2 - 2.0 best for objrec
					"Layer.Learn.TrgAvgAct.TrgRange.Max": "2.0",  // 2.0
				}},
			{Sel: ".Hidden", Desc: "noise!",
				Params: params.Params{
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.005",   // 0.005 > 0.01 probably
					"Layer.Act.Noise.Type": "NoNoise", // probably not needed!
				}},
			{Sel: ".CT", Desc: "",
				Params: params.Params{
					"Layer.CtxtGeGain":      "0.2",
					"Layer.Inhib.Layer.Gi":  "1.1",
					"Layer.Act.KNa.On":      "true",
					"Layer.Act.NMDA.Gbar":   "0.03", // larger not better
					"Layer.Act.GABAB.Gbar":  "0.2",
					"Layer.Act.Decay.Act":   "0.0",
					"Layer.Act.Decay.Glong": "0.0",
				}},
			{Sel: "TRCLayer", Desc: "",
				Params: params.Params{
					"Layer.TRC.DriveScale":  "0.05",
					"Layer.Act.GABAB.Gbar":  "0.005", // 0.005 > 0.01 > 0.002 -- sensitive
					"Layer.Act.NMDA.Gbar":   "0.1",   // 0.1 > .05 > .2
					"Layer.Act.Clamp.Rate":  "180",   // 120 == 100 > 150
					"Layer.Act.Decay.Act":   "0.5",
					"Layer.Act.Decay.Glong": "1", // clear long
				}},
			{Sel: "#V2Pd", Desc: "depth input layers use pool inhibition, weaker global?",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.8", // some weaker global inhib
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.1",
				}},
			{Sel: "#S1V", Desc: "S1V regular",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
				}},
			{Sel: "#MSTdP", Desc: "MT uses pool inhibition, full global?",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1",
				}},
			{Sel: ".cIPL", Desc: "cIPL global",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
				}},
			{Sel: "#cIPLP", Desc: "cIPL global",
				Params: params.Params{
					"Layer.TRC.NoTopo": "true", // true def
				}},
			{Sel: ".PCC", Desc: "PCC uses pool inhibition but is treated as full",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.1",
				}},
			{Sel: "#PCCP", Desc: "no topo",
				Params: params.Params{
					"Layer.TRC.NoTopo": "true", // true def
				}},
			{Sel: "#SMAP", Desc: "no topo",
				Params: params.Params{
					"Layer.TRC.NoTopo": "true", // true def
				}},
			{Sel: "#VL", Desc: "VL regular inhib",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1",
				}},
			{Sel: ".S1S", Desc: "lower inhib, higher act",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1", // some weaker global inhib
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "0.8", // weaker
					"Layer.Inhib.ActAvg.Init": "0.5",
				}},
			{Sel: "#M1", Desc: "noise!",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1", // reg
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.01", // 0.01 orig -- some noise essential for 1 self
					"Layer.Act.Noise.Type": "NoNoise",
				}},
			{Sel: "#SMA", Desc: "noise!",
				Params: params.Params{
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.01", // 0.01 orig
					"Layer.Act.Noise.Type": "NoNoise",
				}},
			{Sel: ".IT", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
				}},
			{Sel: ".LIP", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.1",
				}},

			//////////////////////////////////////////////////////////
			// Prjns

			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.Lrate":        "0.04", // critical for lrate sched
					"Prjn.SWt.Adapt.Lrate":    "0.1",  // 0.01 seems to work fine, but .1 maybe more reliable
					"Prjn.SWt.Adapt.SigGain":  "6",
					"Prjn.SWt.Adapt.DreamVar": "0.0", // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":      "1.0", // .5 ok here, 1 best for larger nets: objrec, lvis
					"Prjn.SWt.Init.Mean":      "0.5", // 0.5 generally good
					"Prjn.SWt.Limit.Min":      "0.2",
					"Prjn.SWt.Limit.Max":      "0.8",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".CTBack", Desc: "deep top-down -- stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.2 > 0.5
				}},
			{Sel: ".Lateral", Desc: "default for lateral -- not using",
				Params: params.Params{
					"Prjn.SWt.Init.Sym":  "false",
					"Prjn.SWt.Init.Var":  "0",
					"Prjn.PrjnScale.Rel": "0.02", // .02 > .05 == .01 > .1  -- very minor diffs on TE cat
				}},
			{Sel: ".CTFmSuper", Desc: "CT from main super -- fixed one2one",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5", // 0.5 > 0.2
				}},
			{Sel: ".SuperFwd", Desc: "standard superficial forward prjns -- not to output",
				Params: params.Params{
					"Prjn.Com.PFail":      "0.0", //
					"Prjn.Com.PFailWtMax": "1.0", // 0.8 default
				}},
			{Sel: ".FmPulv", Desc: "default for pulvinar",
				Params: params.Params{
					"Prjn.PrjnScale.Rel":  "0.1", // .1 > .2
					"Prjn.Com.PFail":      "0.0", // try
					"Prjn.Com.PFailWtMax": "1.0", // 0.8 default
				}},
			{Sel: ".CTSelf", Desc: "CT to CT",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5", // 0.5 > 0.2
				}},
			{Sel: ".FwdToPulv", Desc: "feedforward to pulvinar directly",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
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
		},
	}},
}
