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
			{Sel: "#V2Pd", Desc: "depth input layers use pool inhibition, weaker global?",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.2", // some weaker global inhib
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.8",
				}},
			{Sel: "#S1S", Desc: "S1 uses pool inhib",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.0", // some weaker global inhib
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.4", // weaker
				}},
			{Sel: "#S1V", Desc: "S1V regular",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
			{Sel: "#MSTdP", Desc: "MT uses pool inhibition, full global?",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.6",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.6",
				}},
			{Sel: ".cIPL", Desc: "cIPL global",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
			{Sel: "#cIPLP", Desc: "cIPL global",
				Params: params.Params{
					"Layer.TRC.NoTopo": "true", // true def
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
			{Sel: "#VL", Desc: "VL regular inhib",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.6",
				}},
			{Sel: "#M1", Desc: "noise!",
				Params: params.Params{
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.01", // 0.01 orig -- some noise essential for 1 self
					"Layer.Act.Noise.Type": "GeNoise",
					"Layer.Inhib.Layer.Gi": "1.8", // reg
				}},
			{Sel: "#SMA", Desc: "noise!",
				Params: params.Params{
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.01", // 0.01 orig
					"Layer.Act.Noise.Type": "GeNoise",
				}},
			{Sel: ".IT", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
			{Sel: ".LIP", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.6",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1.6",
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
					"Prjn.WtInit.Mean": "0.5", // 0.8 better for wwi3d, 0.5 default
					"Prjn.WtScale.Rel": "0.5",
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
