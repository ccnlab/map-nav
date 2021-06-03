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
					"Layer.Inhib.ActAvg.Init":            "0.06",
					"Layer.Inhib.ActAvg.Targ":            "0.06",
					"Layer.Inhib.Layer.Gi":               "1.1",
					"Layer.Act.Gbar.L":                   "0.2",
					"Layer.Act.Decay.Act":                "0.0", // todo: explore
					"Layer.Act.Decay.Glong":              "0.0",
					"Layer.Act.Dt.TrlAvgTau":             "20",    // 20 > higher for objrec, lvis
					"Layer.Learn.TrgAvgAct.ErrLrate":     "0.01",  // 0.01 lvis
					"Layer.Learn.TrgAvgAct.SynScaleRate": "0.005", // 0.005 lvis
					"Layer.Learn.TrgAvgAct.TrgRange.Min": "0.5",   // .5 best for Lvis, .2 - 2.0 best for objrec
					"Layer.Learn.TrgAvgAct.TrgRange.Max": "2.0",   // 2.0
				}},
			{Sel: ".Hidden", Desc: "noise? sub-pools",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":    "0.06",
					"Layer.Inhib.ActAvg.Targ":    "0.06",
					"Layer.Inhib.ActAvg.AdaptGi": "true", // enforce for these guys
					"Layer.Inhib.Layer.Gi":       "1.1",
					"Layer.Inhib.Pool.Gi":        "1.1",
					"Layer.Inhib.Pool.On":        "true", // independent pathways
					"Layer.Inhib.Layer.On":       "false",
					"Layer.Act.Decay.Act":        "0.0", // todo: explore
					"Layer.Act.Decay.Glong":      "0.0",
					"Layer.Act.Noise.Dist":       "Gaussian",
					"Layer.Act.Noise.Var":        "0.005",   // 0.005 > 0.01 probably
					"Layer.Act.Noise.Type":       "NoNoise", // probably not needed!
				}},
			{Sel: ".CT", Desc: "",
				Params: params.Params{
					"Layer.CtxtGeGain":      "0.2",
					"Layer.Inhib.Layer.Gi":  "1.1",
					"Layer.Inhib.Pool.Gi":   "1.1",
					"Layer.Inhib.Pool.On":   "true", // independent pathways
					"Layer.Inhib.Layer.On":  "false",
					"Layer.Act.KNa.On":      "true",
					"Layer.Act.NMDA.Gbar":   "0.03", // larger not better
					"Layer.Act.GABAB.Gbar":  "0.2",
					"Layer.Act.Decay.Act":   "0.0", // todo: explore
					"Layer.Act.Decay.Glong": "0.0",
				}},
			{Sel: "TRCLayer", Desc: "",
				Params: params.Params{
					"Layer.TRC.DriveScale":  "0.05",
					"Layer.Act.GABAB.Gbar":  "0.005", // 0.005 > 0.01 > 0.002 -- sensitive
					"Layer.Act.NMDA.Gbar":   "0.1",   // 0.1 > .05 > .2
					"Layer.Act.Decay.Act":   "0.5",
					"Layer.Act.Decay.Glong": "1", // clear long
				}},
			{Sel: ".Depth", Desc: "depth layers use pool inhibition only",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.ActAvg.Targ": "0.12",
					"Layer.Inhib.Layer.On":    "false", // pool only
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1",
				}},
			{Sel: ".Fovea", Desc: "fovea has both",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.11",
					"Layer.Inhib.ActAvg.Targ": "0.11",
					"Layer.Inhib.Layer.On":    "true", // layer too
					"Layer.Inhib.Layer.Gi":    "1",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1",
				}},
			{Sel: ".S1S", Desc: "lower inhib, higher act",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1", // some weaker global inhib
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "0.8", // weaker
					"Layer.Inhib.ActAvg.Init": "0.2",
					"Layer.Inhib.ActAvg.Targ": "0.2",
				}},
			{Sel: ".S1V", Desc: "lower inhib, higher act",
				Params: params.Params{
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.On":     "false",
					"Layer.Inhib.ActAvg.Init": "0.1",
					"Layer.Inhib.ActAvg.Targ": "0.1",
				}},
			{Sel: ".Ins", Desc: "pools",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1.1",
					"Layer.Inhib.ActAvg.Init": "0.1",
					"Layer.Inhib.ActAvg.Targ": "0.1",
				}},
			{Sel: ".cIPL", Desc: "cIPL global",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
				}},
			{Sel: "#cIPLP", Desc: "cIPL global",
				Params: params.Params{
					"Layer.TRC.NoTopo": "true", // true def
				}},
			{Sel: "#PCCP", Desc: "no topo",
				Params: params.Params{
					"Layer.TRC.NoTopo": "true", // true def
				}},
			{Sel: "#SMAP", Desc: "topo -- map M1",
				Params: params.Params{
					"Layer.TRC.NoTopo":        "false", // definitely topo here!
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.Gi":     "1.1",
					"Layer.Inhib.Pool.On":     "true", // independent pathways
					"Layer.Inhib.Layer.On":    "false",
					"Layer.Inhib.ActAvg.Init": "0.1",
					"Layer.Inhib.ActAvg.Targ": "0.1",
				}},
			{Sel: "#VL", Desc: "VL regular inhib",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1",
				}},
			{Sel: "#M1", Desc: "noise!?",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1", // reg
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.01", // 0.01 orig -- some noise essential for 1 self
					"Layer.Act.Noise.Type": "NoNoise",
				}},
			{Sel: "#SMA", Desc: "noise!?",
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
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.1",
				}},

			//////////////////////////////////////////////////////////
			// Prjns

			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":   "0.04", // critical for lrate sched
					"Prjn.SWt.Adapt.Lrate":    "0.01", // 0.01 seems to work fine, but .1 maybe more reliable
					"Prjn.SWt.Adapt.SigGain":  "6",
					"Prjn.SWt.Adapt.DreamVar": "0.01", // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":      "1.0",  // .5 ok here, 1 best for larger nets: objrec, lvis
					"Prjn.SWt.Init.Mean":      "0.5",  // 0.5 generally good
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
			{Sel: ".Inhib", Desc: "inhibitory projection",
				Params: params.Params{
					"Prjn.Learn.Learn":      "true",   // learned decorrel is good
					"Prjn.Learn.Lrate.Base": "0.0001", // .0001 > .001 -- slower better!
					"Prjn.SWt.Init.Var":     "0.0",
					"Prjn.SWt.Init.Mean":    "0.1",
					"Prjn.SWt.Adapt.On":     "false",
					"Prjn.PrjnScale.Init":   "0.1", // .1 = .2, slower blowup
					"Prjn.PrjnScale.Adapt":  "false",
					"Prjn.IncGain":          "1", // .5 def
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
