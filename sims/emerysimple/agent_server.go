package main

import (
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/etable/etensor"
)

// TODO Document the pairing between this class and Socket World. Maybe rename one or both.

type SocketAgentServer struct {
	AgentInterface
	Loops *looper.Manager
	World *SocketWorld `desc:"World represents the World to the agent. It can masquerade as a WorldInterface. It holds actions that the agent has taken, and holds observations for the agent."`
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
	return "" // Return agent name or type or requests for the environment or something.
}

func (agent *SocketAgentServer) Step(observations map[string]etensor.Tensor, debug string) map[string]Action {
	agent.World.CachedObservations = observations // Record observations for this timestep for the world to report.
	agent.Loops.Step(1, etime.Trial)
	// After 1 trial has been stepped, a new action will be ready to return.
	return agent.World.CachedActions
}
