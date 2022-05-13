package main

import (
	"context"

	"github.com/Astera-org/worlds/network"
	"github.com/Astera-org/worlds/network/gengo/env"
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

// this implements the thrift interface and serves as a proxy
// between the network world and the local types
type AgentHandler struct {
	agent *SocketAgentServer
}

func (handler AgentHandler) Init(ctx context.Context, actionSpace env.Space,
	observationSpace env.Space) (string, error) {
	return handler.agent.Init(transformSpace(actionSpace), transformSpace(observationSpace)), nil
}

func (handler AgentHandler) Step(ctx context.Context, observations env.Observations, debug string) (env.Actions, error) {
	obs := transformObservations(observations)
	actions := handler.agent.Step(obs, debug)
	return transformActions(actions), nil
}

func transformActions(actions map[string]Action) env.Actions {
	res := make(env.Actions)
	for k, v := range actions {
		res[k] = toEnvAction(&v)
	}
	return res
}

func transformSpace(space env.Space) map[string]SpaceSpec {
	res := make(map[string]SpaceSpec)
	for k, v := range space {
		res[k] = toSpaceSpec(v)
	}
	return res
}

func transformObservations(observations env.Observations) map[string]etensor.Tensor {
	res := make(map[string]etensor.Tensor)
	for k, v := range observations {
		res[k] = toTensor(v)
	}
	return res
}

func toSpaceSpec(spec *env.SpaceSpec) SpaceSpec {
	return SpaceSpec{
		ContinuousShape: toInt(spec.Shape.Shape),
		Stride:          toInt(spec.Shape.Stride),
		Min:             spec.Min, Max: spec.Max,
		DiscreteLabels: spec.DiscreteLabels,
	}
}

func fromSpaceSpec(spec *SpaceSpec) *env.SpaceSpec {
	return &env.SpaceSpec{
		Shape: &env.Shape{Shape: toInt32(spec.ContinuousShape), Stride: toInt32(spec.Stride)},
		Min:   spec.Min,
		Max:   spec.Max,
	}
}

func toEnvAction(action *Action) *env.Action {
	return &env.Action{
		ActionShape:    fromSpaceSpec(action.ActionShape),
		Vector:         fromTensor(action.Vector),
		DiscreteOption: int32(action.DiscreteOption),
	}
}

func toTensor(envtensor *env.ETensor) etensor.Tensor {
	return etensor.NewFloat64Shape(toShape(envtensor.Shape), envtensor.Values)
}

func toShape(shape *env.Shape) *etensor.Shape {
	return &etensor.Shape{
		Shp:  toInt(shape.Shape),
		Strd: toInt(shape.Stride),
		Nms:  shape.Names,
	}
}

func fromShape(shape *etensor.Shape) *env.Shape {
	return &env.Shape{
		Shape:  toInt32(shape.Shp),
		Stride: toInt32(shape.Strd),
		Names:  shape.Nms,
	}
}

func fromTensor(tensor etensor.Tensor) *env.ETensor {
	res := &env.ETensor{
		Shape:  fromShape(tensor.ShapeObj()),
		Values: nil, // gets set in the next line
	}
	tensor.Floats(&res.Values)
	return res
}

func toInt(xs []int32) []int {
	res := make([]int, len(xs))
	for i := range xs {
		res[i] = int(xs[i])
	}
	return res
}

func toInt32(xs []int) []int32 {
	res := make([]int32, len(xs))
	for i := range xs {
		res[i] = int32(xs[i])
	}
	return res
}

// StartServer blocks, waiting for calls from the environment
func (agent *SocketAgentServer) StartServer() {
	handler := AgentHandler{agent}
	server := network.MakeServer(handler)
	server.Serve()
}

func (agent *SocketAgentServer) Init(actionSpace map[string]SpaceSpec, observationSpace map[string]SpaceSpec) string {
	// TODO If you want, you could add a callback here to reconfigure the network based on the action and observation spaces.
	agent.Loops.Init()
	return "" // Return agent name or type or requests for the environment or something.
}

func (agent *SocketAgentServer) Step(observations map[string]etensor.Tensor, debug string) map[string]Action {
	agent.World.CachedObservations = observations // Record observations for this timestep for the world to report.
	agent.Loops.Step(1, etime.Trial)
	// After 1 trial has been stepped, a new action will be ready to return.
	return agent.World.CachedActions
}
