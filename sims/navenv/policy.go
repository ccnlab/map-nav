// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package navenv

import (
	"github.com/emer/emergent/erand"
	"github.com/goki/gi/mat32"
	"github.com/goki/ki/kit"
)

// Policy provides a parameterized default "subcortical" navigation policy
type Policy struct {
	Auto         bool       `desc:"let the network do the driving, instead of having strong training wheels"`
	UseInAct     float32    `desc:"probability of using input activation from model"`
	AvoidMinDist float32    `desc:"when min dist goes below this value, head turns to find better dir"`
	PBodyTurn    float32    `desc:"probability of turning randomly when otherwise walking freely"`
	AvgDistPTurn float32    `desc:"multiplier on 1 / avg dist for probability of turning when otherwise walking freely"`
	AvoidPGo     float32    `desc:"probability of going when avoid turning if current avg dist > prev"`
	CurState     ActState   `inactive:"+" desc:"current action state"`
	PrvState     ActState   `inactive:"+" desc:"prev action state"`
	PrvAct       Actions    `inactive:"+" desc:"previous action"`
	CurMinDist   float32    `inactive:"+" desc:"current min dist"`
	CurAvgDist   float32    `inactive:"+" desc:"current avg dist"`
	PrvMinDist   float32    `view:"-" inactive:"+" desc:"prev min dist"`
	PrvAvgDist   float32    `view:"-" inactive:"+" desc:"prev avg dist"`
	CurPos       mat32.Vec2 `inactive:"+" desc:"current position"`
}

func (pl *Policy) Defaults() {
	pl.AvoidMinDist = .2
	pl.PBodyTurn = 0.1
	pl.AvgDistPTurn = .05
	pl.AvoidPGo = 0.5
	pl.UseInAct = 0.2
}

// Act is main interface call that updates dist and selects action and updates state
func (pl *Policy) Act(mind, avgd float32, curpos mat32.Vec2, inact Actions) Actions {
	pl.PrvMinDist = pl.CurMinDist
	pl.PrvAvgDist = pl.CurAvgDist
	pl.CurMinDist = mind
	pl.CurAvgDist = avgd
	pl.CurPos = curpos
	var act Actions
	if pl.Auto {
		act = pl.ActChooseAuto(inact)
	} else {
		act = pl.ActChooseTrain(inact)
	}
	pl.PrvAct = act
	// fmt.Printf("chose action: %v  from  in act: %v  min, avg d: %g  %g\n", act, inact, mind, avgd)
	return act
}

// ActChooseTrain makes actual choice based on current state -- called by Act -- for
// extensive training wheels mode
func (pl *Policy) ActChooseTrain(inact Actions) Actions {
	min := float32(0.1)
	max := 1.0 - min
	switch {
	case pl.CurState == NoActState:
		pl.NewState(MovingForward)
		return StepForward
	case pl.CurState == AvoidTurnHead:
		if pl.CurMinDist > pl.AvoidMinDist {
			goo := erand.BoolP(pl.AvoidPGo)
			if goo {
				pl.NewState(MovingForward)
				return StepForward
			}
		}
		return pl.PrvAct // go same dir
	case pl.CurPos.X < min || pl.CurPos.Y < min || pl.CurPos.X > max || pl.CurPos.Y > max:
		pl.NewState(AvoidTurnHead)
		return pl.RndHeadTurn()
	case pl.CurMinDist < pl.AvoidMinDist:
		pl.NewState(AvoidTurnHead)
		return pl.RndHeadTurn()
	case pl.CurState == MovingForward:
		bturn := erand.BoolP(pl.PBodyTurn)
		if bturn {
			return pl.RndBodyTurn()
		}
		dfac := pl.AvgDistPTurn / pl.CurAvgDist
		bturn = erand.BoolP(dfac) // todo: add head turns
		if bturn {
			return pl.RndBodyTurn()
		}
	}
	if erand.BoolP(pl.UseInAct) {
		return inact
	}
	return StepForward
}

// ActChooseAuto makes actual choice based on current state -- called by Act -- for
// autonomous behavior mode -- only act when absolutely necessary
func (pl *Policy) ActChooseAuto(inact Actions) Actions {
	min := float32(0.1)
	max := 1.0 - min
	switch {
	case pl.CurState == NoActState:
		pl.NewState(MovingForward)
		return StepForward
	case pl.CurPos.X < min || pl.CurPos.Y < min || pl.CurPos.X > max || pl.CurPos.Y > max:
		pl.NewState(AvoidTurnHead)
		if pl.CurMinDist < 0.8*pl.AvoidMinDist {
			if inact == RotHeadLeft || inact == RotHeadRight {
				return inact
			}
			return pl.RndHeadTurn()
		}
		return StepForward
	default:
		pl.NewState(MovingForward)
		return inact
	}
}

func (pl *Policy) RndBodyTurn() Actions {
	rt := erand.BoolP(.5)
	if rt {
		return RotBodyRight
	} else {
		return RotBodyLeft
	}
}

func (pl *Policy) RndHeadTurn() Actions {
	rt := erand.BoolP(.5)
	if rt {
		return RotHeadRight
	} else {
		return RotHeadLeft
	}
}

func (pl *Policy) NewState(st ActState) {
	pl.PrvState = pl.CurState
	pl.CurState = st
}

// ActState is action state
type ActState int

//go:generate stringer -type=ActState

var KiT_ActState = kit.Enums.AddEnum(ActStateN, false, nil)

// The action states
const (
	NoActState ActState = iota

	// Moving Forward is just moving forward
	MovingForward

	// AvoidTurnHead is turning head to find a way to avoid obstacle
	AvoidTurnHead

	ActStateN
)
