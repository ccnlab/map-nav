package main

import (
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

// TODO Merge with agent server

// TODO Comments

type SocketWorld struct {
	WorldInterface
	CachedObservations map[string]etensor.Tensor `desc:"Observations from the last step."`
	CachedActions      map[string]Action         `desc:"Actions the action wants to take this step."`
}

// Init sets up a server and waits for the agent to handshake with it for initiation.
func (world *SocketWorld) Init(details string) (map[string]SpaceSpec, map[string]SpaceSpec) {
	// This does nothing. The external world initializes itself.
	return nil, nil // Return action space and observation space.
}

func (world *SocketWorld) Step(actions map[string]Action, agentDone bool) (map[string]etensor.Tensor, bool, string) {
	world.CachedActions = actions
	world.CachedObservations = nil
	return nil, false, "" // Return observations, done, and debug string.
}

func (world *SocketWorld) Observe(name string) etensor.Tensor {
	if world.CachedObservations == nil {
		return nil
	}
	obs, ok := world.CachedObservations[name]
	if ok {
		return obs
	}
	return nil
}

func getRandomTensor(shape SpaceSpec) etensor.Tensor {
	rt := etensor.NewFloat32(shape.ContinuousShape, shape.Stride, nil)
	patgen.PermutedBinaryRows(rt, 1, 1, 0)
	return rt
}

func (world *SocketWorld) ObserveWithShape(name string, shape SpaceSpec) etensor.Tensor {
	// TODO Actually call Observe and reshape it.
	return getRandomTensor(shape)
}
