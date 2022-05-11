package main

import (
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/etable/etensor"
)

type SocketAgentServer struct {
	AgentInterface
	Loops *looper.Manager
}

// StartServer blocks, and sometimes calls Init or Step.
func (agent *SocketAgentServer) StartServer() {
	// TODO Set up server and replace logic below.

	// TODO Replace this logic which does not use the network at all.
	agent.Init(nil, nil)
	for {
		agent.Step(nil, "Everything is fine")
	}
}

func (agent *SocketAgentServer) Init(actionSpace map[string]SpaceSpec, observationSpace map[string]SpaceSpec) string {
	agent.Loops.Init()
	return ""
}

func (agent *SocketAgentServer) Step(observations map[string]etensor.Tensor, debug string) map[string]Action {
	// TODO Pass observations in, get Actions out.
	agent.Loops.Step(1, etime.Trial)
	// After 1 trial has been stepped, a new action will be ready to return.
	return nil // TODO Return action.
}
