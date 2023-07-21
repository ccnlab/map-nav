// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/prjn"
	"github.com/emer/empi/mpi"
)

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {
	Env            map[string]any `desc:"env parameters -- can set any field/subfield on Env struct, using standard TOML formatting"`
	NOutPer        int            `def:"5" desc:"number of units per localist output unit"`
	PctCortexStEpc int            `def:"10" desc:"epoch when PctCortex starts increasing"`
	PctCortexNEpc  int            `def:"1" desc:"number of epochs over which PctCortexMax is reached"`
	PctCortex      float32        `inactive:"+" desc:"proportion of behavioral approach sequences driven by the cortex vs. hard-coded reflexive subcortical"`
	SameSeed       bool           `desc:"for testing, force each env to use same seed"`
}

// CurPctCortex returns current PctCortex and updates field, based on epoch counter
func (cfg *EnvConfig) CurPctCortex(epc int) float32 {
	if epc >= cfg.PctCortexStEpc && cfg.PctCortex < 1 {
		cfg.PctCortex = float32(epc-cfg.PctCortexStEpc) / float32(cfg.PctCortexNEpc)
		if cfg.PctCortex > 1 {
			cfg.PctCortex = 1
		} else {
			mpi.Printf("PctCortex updated to: %g at epoch: %d\n", cfg.PctCortex, epc)
		}
	}
	return cfg.PctCortex
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {
	Network  map[string]any `desc:"network parameters"`
	SubPools bool           `def:"false" desc:"if true, organize layers and connectivity with 2x2 sub-pools within each topological pool"`
	Sheet    string         `desc:"Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params"`
	Tag      string         `desc:"extra tag to add to file names and logs saved from this run"`
	Note     string         `desc:"user note -- describe the run params etc -- like a git commit message for the run"`
	File     string         `nest:"+" desc:"Name of the JSON file to input saved parameters from."`
	SaveAll  bool           `nest:"+" desc:"Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params"`
	Good     bool           `nest:"+" desc:"for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time."`
	Prjns    Prjns          `nest:"+" desc:"special projections"`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {
	MPI         bool `desc:"use mpi message passing interface for distributed data parallel computation"`
	GPU         bool `def:"true" desc:"use the GPU for computation -- generally faster even for small models if NData ~16"`
	NData       int  `def:"16" min:"1" desc:"number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning."`
	NThreads    int  `def:"0" desc:"number of parallel threads for CPU computation -- 0 = use default"`
	Run         int  `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`
	NRuns       int  `def:"5" min:"1" desc:"total number of runs to do when running Train"`
	NEpochs     int  `def:"150" desc:"total number of epochs per run"`
	NTstEpochs  int  `def:"200" desc:"total number of testing epochs per run"`
	NTrials     int  `def:"128" desc:"total number of trials per epoch.  Should be an even multiple of NData."`
	PCAInterval int  `def:"10" desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {
	RFTargs []string `def:"['Pos','Act','HdDir']" desc:"special targets for activation-based receptive field maps"`
	SaveWts bool     `desc:"if true, save final weights after each run"`
	Epoch   bool     `def:"true" nest:"+" desc:"if true, save train epoch log to file, as .epc.tsv typically"`
	Run     bool     `def:"true" nest:"+" desc:"if true, save run log to file, as .run.tsv typically"`
	Trial   bool     `def:"false" nest:"+" desc:"if true, save train trial log to file, as .trl.tsv typically. May be large."`
	NetData bool     `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {
	Includes []string    `desc:"specify include files here, and after configuration, it contains list of include files added"`
	GUI      bool        `def:"true" desc:"open the GUI -- does not automatically run -- if false, then runs automatically and quits"`
	Debug    bool        `desc:"log debugging information"`
	Env      EnvConfig   `view:"add-fields" desc:"environment configuration options"`
	Params   ParamConfig `view:"add-fields" desc:"parameter related configuration options"`
	Run      RunConfig   `view:"add-fields" desc:"sim running related configuration options"`
	Log      LogConfig   `view:"add-fields" desc:"data logging related configuration options"`
}

func (cfg *Config) Defaults() {
	cfg.Params.Prjns.Defaults()
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }

/////////////////////////////////////////////////////
//   Prjns

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

func (pj *Prjns) Defaults() {
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