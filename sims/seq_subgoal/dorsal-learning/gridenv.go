// Copyright (c) 2019, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	// "image"
	"fmt"
	"io/ioutil"
	"math/rand"

	// "log"
	"strings"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/popcode"
	"github.com/goki/mat32"

	// "github.com/emer/emergent/erand"
	// "github.com/emer/emergent/popcode"
	"github.com/emer/etable/etensor"
	// "github.com/emer/etable/tsragg"
	// "github.com/emer/eve/eve"
	// "github.com/emer/eve/evev"
	// "github.com/goki/gi/gi"
	// "github.com/goki/gi/gi3d"
	// "github.com/goki/gi/giv"
	// "github.com/goki/gi/oswin"
	// "github.com/goki/gi/oswin/gpu"
	// "github.com/goki/gi/units"
	// "github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// Actions is a list of available actions for model
type Actions int

//go:generate stringer -type=Actions

var KiT_Actions = kit.Enums.AddEnum(ActionsN, false, nil)

// The actions avail
const (
	// forward implies rot body to head
	North Actions = iota
	East
	South
	West
	ActionsN
)

// ActionsCode are code letters for the actions
var ActionsCode = []string{"N", "E", "S", "W"}

type Pos struct {
	Row int
	Col int
}
type Tile struct {
	open    bool
	color   int
	visited bool
}

var tilekey = map[rune]Tile{
	' ': Tile{open: true},
	'#': Tile{open: false},
}

func (env *Env) NewTile(symbol rune) Tile {
	tile := tilekey[symbol]
	tile.color = rand.Intn(env.Colors)
	return tile

}

type World struct {
	grid [][]Tile
}

func (env *Env) NewWorld(filename string) World {
	dat, err := ioutil.ReadFile(filename)

	if err != nil {
		panic(err)
	}
	datstr := string(dat)
	datstr = strings.Trim(datstr, "\n")
	lines := strings.Split(datstr, "\n")

	world := World{
		grid: make([][]Tile, len(lines)),
	}
	for i := range lines {
		world.grid[i] = make([]Tile, len(lines[0]))
	}
	for row, line := range lines {
		for col, symbol := range line {
			world.grid[row][col] = env.NewTile(symbol)
		}
	}
	return world

}

func (wld *World) Loc(pos Pos) *Tile {
	return &wld.grid[pos.Row][pos.Col]
}

// Env manages the navigation environment
type Env struct {
	Nm         string  `desc:"name of this environment"`
	Dsc        string  `desc:"description of this environment"`
	Run        env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr `view:"inline" desc:"number of times through arbitrary number of Events"`
	Event      env.Ctr `view:"inline" desc:"current ordinal item in Table -- if Sequential then = row number in table, otherwise is index in Order list that then gives row number in Table"`
	CurPos     Pos     `desc:"current normalized position"`
	PosRes     int
	CurPosMap  etensor.Float32 `desc:"current position as 1-hot tensor"`
	ActRes     int
	CurAct     Actions         `desc:"current action selected"`
	PrvAct     Actions         `desc:"previous action selected"`
	ExtAct     Actions         `desc:"current externally-supplied action"`
	CurActMap  etensor.Float32 `desc:"action as a 1 hot tensor, returned as state"`
	PrvActMap  etensor.Float32 `desc:"action as a 1 hot tensor, returned as state"`
	ColorMap   etensor.Float32 `desc:" color as a 1 hot tensor, returned as state"`
	World      World
	Policy     Policy
	PrevPos    Pos
	NextPosMap etensor.Float32
	OffCycle   bool `desc:"toggled each cycle to only update every other cycle, allowing the network a cycle to develop predictions"`
	Colors     int
	pop2D      popcode.TwoD
}

func (ev *Env) Name() string { return ev.Nm }
func (ev *Env) Desc() string { return ev.Dsc }

// String returns the current state as a string
func (ev *Env) String() string {
	return fmt.Sprintf("Run %d Epoch %d Event %d ", ev.Run.Cur, ev.Epoch.Cur, ev.Event.Cur)
}
func (ev *Env) Validate() error {
	return nil
}

func (ev *Env) Defaults() {

}

func (ev *Env) Init(run int) {

	ev.Colors = 20
	ev.MakeWorld()
	rows, cols := len(ev.World.grid), len(ev.World.grid[0])
	ev.CenterAgent()
	ev.PrevPos = ev.CurPos

	ev.pop2D = popcode.TwoD{}
	ev.pop2D.Code = popcode.GaussBump
	ev.pop2D.Min.Set(0.0, 0.0)
	ev.pop2D.Max.Set(float32(rows), float32(cols))
	sigma := float32(0.001)
	ev.pop2D.Sigma.Set(sigma, sigma)
	ev.pop2D.Thr = 0.1
	ev.pop2D.Clip = true
	ev.pop2D.MinSum = 0.2

	ev.Policy.Defaults()

	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Event.Scale = env.Event
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Event.Init()
	ev.ActRes = 1
	ev.PosRes = 1
	ev.Run.Cur = run
	ev.Event.Cur = -1 // init state -- key so that first Step() = 0
	ev.CurPosMap.SetShape([]int{rows, cols}, nil, []string{"Y", "X"})
	ev.CurActMap.SetShape([]int{int(ActionsN), 1, 1, ev.ActRes}, nil, []string{"Action", "1", "1", "1"})
	ev.PrvActMap.SetShape([]int{int(ActionsN), 1, 1, ev.ActRes}, nil, []string{"Action", "1", "1", "1"})
	ev.ColorMap.SetShape([]int{ev.Colors, 1, 1, 1}, nil, []string{"Color", "1", "1", "1"})

	ev.NextPosMap.SetShape([]int{rows, cols}, nil, []string{"Y", "X"})
}

func (ev *Env) CenterAgent() {

	rows, cols := len(ev.World.grid), len(ev.World.grid[0])
	ev.CurPos.Row = rows / 2
	ev.CurPos.Col = cols / 2
}

func (ev *Env) Step() bool {
	// if ev.OffCycle {
	// 	ev.OffCycle = false
	// 	return true
	// }
	ev.OffCycle = true
	// set CurAct to the output of the Policy. ExtAct is the action selected by the network
	ev.CurAct = ev.Policy.Act(ev.ExtAct, ev)
	ev.TakeAction(ev.PrvAct)
	ev.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ev.Event.Incr() { // if true, hit max, reset to 0
		ev.Epoch.Incr()
	}

	return true
}

func (ev *Env) States() env.Elements {
	els := env.Elements{
		{"PosMap", []int{len(ev.World.grid), len(ev.World.grid[0])}, []string{"Y", "X"}},
		{"NextPosMap", []int{len(ev.World.grid), len(ev.World.grid[0])}, []string{"Y", "X"}},
		{"ActMap", []int{int(ActionsN)}, []string{"ActionsN"}},
		{"PrvActMap", []int{int(ActionsN)}, []string{"ActionsN"}},
		{"ColorMap", []int{ev.Colors}, []string{"ColorsN"}},
	}
	return els
}
func (ev *Env) State(element string) etensor.Tensor {
	switch element {
	case "PosMap":
		return &ev.CurPosMap
	case "NextPosMap":
		return &ev.NextPosMap
	case "ActMap":
		return &ev.CurActMap
	case "PrvActMap":
		return &ev.PrvActMap
	case "ColorMap":
		return &ev.ColorMap
	default:
		return nil
	}
}

func (ev *Env) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Event}
}

func (ev *Env) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Event:
		return ev.Event.Query()
	}
	return -1, -1, false
}

func (ev *Env) Actions() env.Elements {
	els := env.Elements{
		{"Action", []int{1}, nil},
	}
	return els
}

// Action just takes the input as either a string or float -> int
// representation of action, with only 1 element (no parameters)
func (ev *Env) Action(element string, input etensor.Tensor) {
	ev.PrvAct = ev.CurAct
}

// SetAction is easier non-standard interface just for this -- mag is 0..1 normalized magnitude of action
func (ev *Env) SetAction(act Actions) {
	ev.PrvAct = ev.CurAct
	ev.ExtAct = act
	ev.CurAct = act
}

// MakeWorld constructs a new virtual physics world
func (ev *Env) MakeWorld() {

	// ev.World = ev.NewWorld("3x3.world")
	ev.World = ev.NewWorld("5X5.world")
	// ev.World = ev.NewWorld("10X10.world")
	// ev.World = ev.NewWorld("16x16.world")
}

// InitWorld does init on world and re-syncs
func (ev *Env) InitWorld() {
}

// ReMakeWorld rebuilds the world and re-syncs with gui
func (ev *Env) ReMakeWorld() {
	ev.MakeWorld()
}

func Move(pos Pos, act Actions) Pos {

	switch act {
	case North:
		newpos := pos
		newpos.Row++
		return newpos
	case East:
		newpos := pos
		newpos.Col++
		return newpos
	case South:
		newpos := pos
		newpos.Row--
		return newpos
	case West:
		newpos := pos
		newpos.Col--
		return newpos
	default:
		return pos
	}

}

// TakeAction implements given action
func (ev *Env) TakeAction(act Actions) {
	//move
	newpos := Move(ev.CurPos, act)
	if ev.World.Loc(newpos).open {
		ev.PrevPos = ev.CurPos
		ev.CurPos = newpos
	}
	ev.World.Loc(newpos).visited = true

	// handle any non position side effects of action
	// update "neural" reps of state
	ev.UpdateState()
}

// UpdateWorld updates world after action
func (ev *Env) UpdateWorld() {
}

// UpdateState updates the current state representations (depth, action)
func (ev *Env) UpdateState() {
	vec := mat32.Vec2{
		X: float32(ev.CurPos.Col),
		Y: float32(ev.CurPos.Row),
	}

	ev.pop2D.Encode(&ev.CurPosMap, vec)

	nextpos := Move(ev.CurPos, Actions(ev.CurAct))
	vec = mat32.Vec2{
		X: float32(nextpos.Col),
		Y: float32(nextpos.Row),
	}

	ev.pop2D.Encode(&ev.NextPosMap, vec)

	ev.CurActMap.SetZeros()
	ev.CurActMap.SetFloat1D(int(ev.CurAct), 1.0)
	ev.PrvActMap.SetZeros()
	ev.PrvActMap.SetFloat1D(int(ev.PrvAct), 1.0)

	curcolor := ev.World.Loc(ev.CurPos).color
	ev.ColorMap.SetZeros()
	ev.ColorMap.SetFloat1D(curcolor, 1.0)

}
