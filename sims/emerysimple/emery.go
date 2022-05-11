// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO Move to Astera-org/models

package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"log"
	"math/rand"
)

func main() {
	TheSim.New()
	TheSim.ConfigEnv()
	TheSim.ConfigNet(TheSim.Net)
	TheSim.Init() // Call after ConfigNet because it applies params.
	TheSim.ConfigLoops()

	userInterface := UserInterface{
		GUI:           &TheSim.GUI,
		StructForView: &TheSim,
		Looper:        TheSim.Loops,
		Network:       TheSim.Net.EmerNet,
		AppName:       "Agent",
		AppTitle:      "Simple Agent",
		AppAbout:      `A simple agent that can handle an arbitrary world.`,
		InitCallback:  TheSim.Init,
	}
	userInterface.CreateAndRunGuiWithAdditionalConfig(
		// This function is only necessary if you want the network to exist in a separate thread, and you want the agent to provide a server that serves intelligent actions. It adds a button to start the server.
		func() {
			userInterface.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Start Server", Icon: "play",
				Tooltip: "Start a server.",
				Active:  egui.ActiveStopped,
				Func: func() {
					userInterface.GUI.IsRunning = true
					userInterface.GUI.ToolBar.UpdateActions() // Disable GUI
					go func() {
						server := SocketAgentServer{
							Loops: userInterface.Looper,
							World: TheSim.WorldEnv.(*SocketWorld),
						}
						server.StartServer()        // The server probably runs forever.
						userInterface.GUI.Stopped() // Reenable GUI
					}()
				},
			})
		})
	// CreateAndRunGui blocks, so don't put any code after this.
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around.
type Sim struct { // TODO(refactor): Remove a lot of this stuff
	Net      *deep.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	GUI      egui.GUI        `view:"-" desc:"manages all the gui elements"`
	Loops    *looper.Manager `view:"no-inline" desc:"contains looper control loops for running sim"`
	Params   params.Sets     `view:"no-inline" desc:"full collection of param sets"`
	ParamSet string          `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	// TODO Switch back to interface.
	WorldEnv WorldInterface `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time     axon.Time      `desc:"axon timing parameters and state"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() { // TODO(refactor): Remove a lot
	ss.Net = &deep.Network{}

	ss.Time.Defaults()

	// see params_def.go for default params
	ss.Params = ParamSets
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

func (ss *Sim) ConfigEnv() {
	ss.WorldEnv = &SocketWorld{}
}

func (ss *Sim) ConfigNet(net *deep.Network) {
	// A simple network for demonstration purposes.
	net.InitName(net, "Emery")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", 10, 10, emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", 10, 10, emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)
	full := prjn.NewFull()
	net.ConnectLayers(inp, hid1, full, emer.Forward)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	// TODO(refactor): why isn't all this in a function?
	net.Defaults()
	SetParams("Network", true, ss.Net, &ss.Params, ss.ParamSet, ss) // only set Network params
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
	rand.Seed(1)
	SetParams("", true, ss.Net, &ss.Params, ss.ParamSet, ss) // all sheets
}

func (ss *Sim) NewRun() {
	ss.Net.InitWts()
}

// TODO Move to library.
func (ss *Sim) AddDefaultLoopSimLogic(manager *looper.Manager) {
	// Net Cycle
	for m, _ := range manager.Stacks {
		manager.Stacks[m].Loops[etime.Cycle].Main.Add("Axon:Cycle:RunAndIncrement", func() {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
		})
	}
	// Weight updates.
	// Note that the substring "UpdateNetView" in the name is important here, because it's checked in AddDefaultGUICallbacks.
	manager.GetLoop(etime.Train, etime.Trial).OnEnd.Add("Axon:LoopSegment:UpdateWeights", func() {
		ss.Net.DWt(&ss.Time)
		// TODO Need to update net view here to accurately display weight changes.
		ss.Net.WtFmDWt(&ss.Time)
	})

	//todo is this neccesary
	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range manager.Stacks {
		curMode := m // For closures.
		for t, loop := range loops.Loops {
			curTime := t
			loop.OnStart.Add(curMode.String()+":"+curTime.String()+":"+"SetTimeVal", func() {
				ss.Time.Mode = curMode.String()
			})
		}
	}
}

// ConfigLoops configures the control loops
func (ss *Sim) ConfigLoops() {
	manager := looper.Manager{}.Init()
	manager.Stacks[etime.Train] = &looper.Stack{}
	manager.Stacks[etime.Train].Init().AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, 2).AddTime(etime.Cycle, 200)
	minusPhase := looper.Event{Name: "MinusPhase", AtCtr: 0}
	minusPhase.OnEvent.Add("Sim:MinusPhase:Start", func() {
		ss.Time.PlusPhase = false
		ss.Time.NewPhase(false)
	})
	plusPhase := looper.Event{Name: "PlusPhase", AtCtr: 150}
	plusPhase.OnEvent.Add("Sim:MinusPhase:End", func() { ss.Net.MinusPhase(&ss.Time) })

	plusPhase.OnEvent.Add("Sim:PlusPhase:Start", func() {
		ss.Time.PlusPhase = true
		ss.Time.NewPhase(true)
	})

	plusPhase.OnEvent.Add("Sim:PlusPhase:SendActionsThenStep", func() {
		ss.SendActionAndStep(ss.Net, ss.WorldEnv) //TODO shouldn't this be called at the END of the plus phase?
	})

	plusPhaseEnd := looper.Event{Name: "PlusPhase", AtCtr: 199}
	plusPhaseEnd.OnEvent.Add("Sim:PlusPhase:End", func() { ss.Net.PlusPhase(&ss.Time) })
	// Add both to train and test, by copy
	manager.AddEventAllModes(etime.Cycle, minusPhase)
	manager.AddEventAllModes(etime.Cycle, plusPhase)
	manager.AddEventAllModes(etime.Cycle, plusPhaseEnd)

	// Trial Stats and Apply Input
	mode := etime.Train // For closures
	stack := manager.Stacks[mode]
	stack.Loops[etime.Trial].OnStart.Add("Sim:ResetState", func() {
		ss.Net.NewState()
		ss.Time.NewState(mode.String())
	})
	stack.Loops[etime.Trial].OnStart.Add("Sim:Trial:Observe",
		func() {
			layers := []string{"Input"}
			ApplyInputsWithStrideAndShape(ss.Net, ss.WorldEnv, layers, layers)
		})

	stack.Loops[etime.Trial].OnEnd.Add("Sim:Trial:QuickScore",
		func() { //
			//loss := ss.Net.LayerByName("VL").(axon.AxonLayer).AsAxon().PctUnitErr()
			//s := fmt.Sprintf("%f", loss)
			//fmt.Println("the pctuniterror is " + s)
		}) //todo put backin
	//stack.Loops[etime.Trial].OnEnd.Add("Sim:StatCounters", ss.StatCounters) //todo put backin
	//stack.Loops[etime.Trial].OnEnd.Add("Sim:TrialStats", ss.TrialStats) //todo put backin

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", ss.NewRun)
	ss.AddDefaultLoopSimLogic(manager)

	// Initialize and print loop structure, then add to Sim
	manager.Init()
	fmt.Println(manager.DocString())
	ss.Loops = manager
}

// SendAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) SendActionAndStep(net *deep.Network, ev WorldInterface) {
	// Get the first Target (output) layer
	ly := net.LayerByName(net.LayersByClass(emer.Target.String())[0]).(axon.AxonLayer).AsAxon()
	//ly := net.LayerByName("VL").(axon.AxonLayer).AsAxon()
	vt := &etensor.Float32{} //ValsTsr(&ss.ValsTsrs, "VL") // TODO Hopefully this doesn't crash
	ly.UnitValsTensor(vt, "ActM")
	//ev.DecodeAndTakeAction("action", vt)
	actions := map[string]Action{"action": Action{Vector: vt}}
	_, _, debug := ev.Step(actions, false)
	if debug != "" {
		fmt.Println("Got debug from Step: " + debug)
	}
}
