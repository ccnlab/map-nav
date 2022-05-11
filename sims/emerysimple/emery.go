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
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"log"
)

func main() {
	var sim Sim
	sim.ConfigEnv()
	sim.Net = sim.ConfigNet()
	sim.ConfigLoops()

	userInterface := UserInterface{
		GUI:           &sim.GUI,
		StructForView: &sim,
		Looper:        sim.Loops,
		Network:       sim.Net.EmerNet,
		AppName:       "Agent",
		AppTitle:      "Simple Agent",
		AppAbout:      `A simple agent that can handle an arbitrary world.`,
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
							World: sim.WorldEnv.(*SocketWorld),
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
	WorldEnv WorldInterface  `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time     axon.Time       `desc:"axon timing parameters and state"`
}

func (ss *Sim) ConfigEnv() {
	ss.WorldEnv = &SocketWorld{}
}

func (ss *Sim) ConfigNet() *deep.Network {
	// A simple network for demonstration purposes.
	net := &deep.Network{}
	net.InitName(net, "Emery")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", 10, 10, emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", 10, 10, emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)
	full := prjn.NewFull()
	net.ConnectLayers(inp, hid1, full, emer.Forward)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	net.Defaults()
	// see params_def.go for default params
	SetParams("Network", true, net, &ParamSets, "", ss)
	err := net.Build()
	if err != nil {
		log.Println(err)
		return nil
	}
	net.InitWts()
	return net
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

func (ss *Sim) NewRun() {
	ss.Net.InitWts()
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

// SendActionAndStep takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) SendActionAndStep(net *deep.Network, ev WorldInterface) {
	// Get the first Target (output) layer
	ly := net.LayerByName(net.LayersByClass(emer.Target.String())[0]).(axon.AxonLayer).AsAxon()
	vt := &etensor.Float32{}
	ly.UnitValsTensor(vt, "ActM")
	//ev.DecodeAndTakeAction("action", vt)
	actions := map[string]Action{"action": {Vector: vt}}
	_, _, debug := ev.Step(actions, false)
	if debug != "" {
		fmt.Println("Got debug from Step: " + debug)
	}
}
