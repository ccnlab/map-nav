// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/popcode"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/metric"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// FWorld is a flat-world grid-based environment
type FWorld struct {
	Nm            string                      `desc:"name of this environment"`
	Dsc           string                      `desc:"description of this environment"`
	Disp          bool                        `desc:"update display -- turn off to make it faster"`
	Size          evec.Vec2i                  `desc:"size of 2D world"`
	PatSize       evec.Vec2i                  `desc:"size of patterns for mats, acts"`
	NYReps        int                         `desc:"number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets"`
	World         *etensor.Int                `view:"no-inline" desc:"2D grid world, each cell is a material (mat)"`
	Mats          []string                    `desc:"list of materials in the world, 0 = empty.  Any superpositions of states (e.g., CoveredFood) need to be discretely encoded, can be transformed through action rules"`
	MatColors     []string                    `desc:"list of material colors"`
	MatMap        map[string]int              `desc:"map of material name to index stored in world cell"`
	BarrierIdx    int                         `desc:"index of material below which (inclusive) cannot move -- e.g., 1 for wall"`
	MatsUSStart   int                         `desc:"index of material where USs start"`
	Pats          map[string]*etensor.Float32 `desc:"patterns for each material (must include Empty) and for each action"`
	ActPats       map[string]*etensor.Float32 `desc:"patterns for each action -- for decoding"`
	Acts          []string                    `desc:"list of actions: starts with: Stay, Left, Right, Forward, Back, then extensible"`
	ActMap        map[string]int              `desc:"action map of action names to indexes"`
	PosUSs        []string                    `desc:"positive USs"`
	NegUSs        []string                    `desc:"negative USs"`
	NDrives       int                         `desc:"number of PosUSs"`
	USMap         map[string]int              `desc:"map of US names to indexes, for both pos and neg"`
	Params        map[string]float32          `desc:"map of optional interoceptive and world-dynamic parameters -- cleaner to store in a map"`
	FOV           int                         `desc:"field of view in degrees, e.g., 180, must be even multiple of VisAngInc"`
	VisAngInc     int                         `desc:"visual angle increment for rotation, in degrees -- defaults to 45"`
	MotAngInc     int                         `desc:"motion angle increment for rotation, in degrees -- defaults to 15"`
	NMotAngles    int                         `inactive:"+" desc:"total number of motion rotation angles in a circle"`
	NFOVRays      int                         `inactive:"+" desc:"total number of FOV rays that are traced"`
	WallUrgency   float32                     `desc:"urgency when right against a wall"`
	EatUrgency    float32                     `desc:"urgency for eating and drinking"`
	CloseUrgency  float32                     `desc:"urgency for being close to food / water"`
	FwdMargin     float32                     `desc:"forward action must be this factor larger than 2nd best option to be selected"`
	ShowRays      bool                        `desc:"for debugging only: show the main depth rays as they are traced out from point"`
	ShowFovRays   bool                        `desc:"for debugging only: show the fovea rays as they are traced out from point"`
	TraceInstinct bool                        `desc:"for debugging, print out a trace of the action generation logic"`
	TraceInst     string                      `inactive:"+" desc:"trace of instinct"`
	FoveaSize     int                         `desc:"number of items on each size of the fovea, in addition to center (0 or more)"`
	FoveaAngInc   int                         `desc:"scan angle for fovea"`
	PopSize       int                         `inactive:"+" desc:"number of units in population codes"`
	PopCode       popcode.OneD                `desc:"generic population code values, in normalized units"`
	DepthSize     int                         `inactive:"+" desc:"number of units in depth population codes"`
	DepthCode     popcode.OneD                `desc:"population code for depth, in normalized units"`
	AngCode       popcode.Ring                `desc:"angle population code values, in normalized units"`

	// current state below (params above)
	PosF          mat32.Vec2                  `inactive:"+" desc:"current location of agent, floating point"`
	PosI          evec.Vec2i                  `inactive:"+" desc:"current location of agent, integer"`
	HeadDir       int                         `inactive:"+" desc:"current angle, in degrees"`
	RotAng        int                         `inactive:"+" desc:"angle that we just rotated -- drives vestibular"`
	Urgency       float32                     `inactive:"+" desc:"for ActGen, level of urgency for following the generated action"`
	LastAct       int                         `inactive:"+" desc:"last action taken"`
	LastEffort    float32                     `inactive:"+" desc:"effort associated with last action taken"`
	ShouldGate    bool                        `inactive:"+" desc:"true if fovea includes a positive CS"`
	JustGated     bool                        `inactive:"+" desc:"just gated on this trial"`
	HasGated      bool                        `inactive:"+" desc:"has gated at some point during sequence"`
	Depths        []float32                   `desc:"depth for each angle (NFOVRays), raw"`
	DepthLogs     []float32                   `desc:"depth for each angle (NFOVRays), normalized log"`
	ViewMats      []int                       `inactive:"+" desc:"material at each angle"`
	FovMats       []int                       `desc:"materials at fovea, L-R"`
	FovDepths     []float32                   `desc:"raw depths to foveal materials, L-R"`
	FovDepthLogs  []float32                   `desc:"normalized log depths to foveal materials, L-R"`
	ProxMats      []int                       `desc:"material at each right angle: front, left, right back"`
	ProxPos       []evec.Vec2i                `desc:"coordinates for proximal grid points: front, left, right, back"`
	CurStates     map[string]*etensor.Float32 `desc:"current rendered state tensors -- extensible map"`
	NextStates    map[string]*etensor.Float32 `desc:"next rendered state tensors -- updated from actions"`
	RefreshEvents map[int]*WEvent             `desc:"list of events, key is tick step, to check each step to drive refresh of consumables -- removed from this active list when complete"`
	AllEvents     map[int]*WEvent             `desc:"list of all events, key is tick step"`
	Run           env.Ctr                     `view:"inline" desc:"current run of model as provided during Init"`
	Epoch         env.Ctr                     `view:"inline" desc:"increments over arbitrary fixed number of trials, for general stats-tracking"`
	Trial         env.Ctr                     `view:"inline" desc:"increments for each step of world, loops over epochs -- for general stats-tracking independent of env state"`
	Tick          env.Ctr                     `view:"monolithic time counter -- counts up time every step -- used for refreshing world state"`
	Event         env.Ctr                     `view:"arbitrary counter for steps within a scene -- resets at consumption event"`
	Scene         env.Ctr                     `view:"arbitrary counter incrementing over a coherent sequence of events: e.g., approaching food -- increments at consumption"`
	Episode       env.Ctr                     `view:"arbitrary counter incrementing over scenes within larger episode: feeding, drinking, exploring, etc"`

	// GUI below
	Trace     *etensor.Int       `view:"no-inline" desc:"trace of movement for visualization"`
	TraceView *etview.TensorGrid `view:"no-inline" desc:"view of the activity trace"`
	WorldView *etview.TensorGrid `view:"no-inline" desc:"view of the world"`
	WorldWin  *gi.Window         `view:"-" desc:"FWorld GUI window"`
	WorldTabs *gi.TabView        `view:"-" desc:"FWorld TabView"`
	IsRunning bool               `view:"-" desc:"FWorld is running"`
}

var KiT_FWorld = kit.Types.AddType(&FWorld{}, FWorldProps)

func (ev *FWorld) Name() string { return ev.Nm }
func (ev *FWorld) Desc() string { return ev.Dsc }

// Config configures the world
func (ev *FWorld) Config(ntrls int) {
	ev.Nm = "Demo"
	ev.Dsc = "Example world with basic actions"
	ev.Mats = []string{"Empty", "Wall"}
	ev.BarrierIdx = 1
	ev.Acts = []string{"Forward", "Left", "Right", "Consume", "None"} // "Stay", "Backward",
	ev.PosUSs = []string{"Water", "Protein", "Sugar", "Salt"}
	ev.NegUSs = []string{"Bump"}
	ev.NDrives = len(ev.PosUSs)

	ev.MatsUSStart = len(ev.Mats)
	for _, us := range ev.PosUSs {
		ev.Mats = append(ev.Mats, us)
	}
	for _, us := range ev.PosUSs {
		ev.Mats = append(ev.Mats, us+"Was")
	}
	ev.MatColors = []string{"lightgrey", "black", "blue", "orange", "red", "white", "navy", "brown", "pink", "gray"}

	ev.Params = make(map[string]float32)

	ev.Params["MoveEffort"] = 1
	ev.Params["RotEffort"] = 1
	ev.Params["BumpPain"] = 0.1
	ev.Params["EnvRefresh"] = 100 // time steps before consumed items are refreshed

	ev.Disp = true
	ev.Size.Set(50, 50)
	ev.PatSize.Set(5, 5)
	ev.NYReps = 4
	ev.VisAngInc = 45
	ev.MotAngInc = 15
	ev.FOV = 180
	ev.FoveaSize = 1
	ev.FoveaAngInc = 5
	ev.WallUrgency = .9
	ev.EatUrgency = .8
	ev.CloseUrgency = .5
	ev.FwdMargin = 2
	ev.PopSize = 16
	ev.PopCode.Defaults()
	ev.PopCode.SetRange(-0.2, 1.2, 0.1)
	popSigma := float32(0.1)
	ev.PopCode.Sigma = popSigma
	ev.DepthSize = 16
	ev.DepthCode.Defaults()
	ev.DepthCode.SetRange(0.1, 1, 0.05)
	ev.DepthCode.Sigma = popSigma
	ev.AngCode.Defaults()
	ev.AngCode.SetRange(0, 1, 0.1)
	ev.AngCode.Sigma = popSigma

	// debugging options:
	ev.ShowRays = false
	ev.ShowFovRays = false
	ev.TraceInstinct = true

	ev.Trial.Max = ntrls

	ev.ConfigPats()
	ev.ConfigImpl()

	// uncomment to generate a new world
	ev.GenWorld()
	fnm := fmt.Sprintf("world_%d.tsv", mpi.WorldRank())
	ev.SaveWorld(gi.FileName(fnm))
}

// ConfigPats configures the bit pattern representations of mats and acts
func (ev *FWorld) ConfigPats() {
	ev.Pats = make(map[string]*etensor.Float32)
	ev.ActPats = make(map[string]*etensor.Float32)
	for _, m := range ev.Mats {
		t := &etensor.Float32{}
		t.SetShape([]int{ev.PatSize.Y, ev.PatSize.X}, nil, []string{"Y", "X"})
		ev.Pats[m] = t
	}
	for _, a := range ev.Acts {
		t := &etensor.Float32{}
		t.SetShape([]int{ev.PatSize.Y, ev.PatSize.X}, nil, []string{"Y", "X"})
		ev.Pats[a] = t
	}
	ev.OpenPats("pats.json") // hand crafted..
	for _, a := range ev.Acts {
		ev.ActPats[a] = ev.Pats[a]
	}
}

// ConfigImpl does the automatic parts of configuration
// generally does not require editing
func (ev *FWorld) ConfigImpl() {
	ev.NFOVRays = (ev.FOV / ev.VisAngInc) + 1
	ev.NMotAngles = (360 / ev.MotAngInc) + 1

	ev.World = &etensor.Int{}
	ev.World.SetShape([]int{ev.Size.Y, ev.Size.X}, nil, []string{"Y", "X"})

	ev.ProxMats = make([]int, 4)
	ev.ProxPos = make([]evec.Vec2i, 4)

	ev.CurStates = make(map[string]*etensor.Float32)
	ev.NextStates = make(map[string]*etensor.Float32)

	dv := etensor.NewFloat32([]int{1, ev.NFOVRays, ev.DepthSize, 1}, nil, []string{"1", "Angle", "Pop", "1"})
	ev.NextStates["Depth"] = dv

	ev.Depths = make([]float32, ev.NFOVRays)
	ev.DepthLogs = make([]float32, ev.NFOVRays)
	ev.ViewMats = make([]int, ev.NFOVRays)

	fsz := 1 + 2*ev.FoveaSize
	fd := etensor.NewFloat32([]int{1, fsz, ev.DepthSize, 1}, nil, []string{"1", "Angle", "Pop", "1"})
	ev.NextStates["FovDepth"] = fd

	fv := etensor.NewFloat32([]int{1, fsz, ev.PatSize.Y, ev.PatSize.X}, nil, []string{"1", "Angle", "Y", "X"})
	ev.NextStates["Fovea"] = fv

	ps := etensor.NewFloat32([]int{1, 4, 2, 1}, nil, []string{"1", "Pos", "OnOff", "1"})
	ev.NextStates["ProxSoma"] = ps

	hd := etensor.NewFloat32([]int{1, ev.PopSize}, nil, []string{"1", "PopAngle"})
	ev.NextStates["HeadDir"] = hd

	av := etensor.NewFloat32([]int{ev.PatSize.Y, ev.PatSize.X}, nil, []string{"Y", "X"})
	ev.NextStates["Action"] = av

	pus := etensor.NewFloat32([]int{len(ev.PosUSs)}, nil, []string{"USs"})
	ev.NextStates["PosUSs"] = pus

	nus := etensor.NewFloat32([]int{len(ev.NegUSs)}, nil, []string{"USs"})
	ev.NextStates["NegUSs"] = nus

	ev.CopyNextToCur() // get CurStates from NextStates

	ev.FovMats = make([]int, fsz)
	ev.FovDepths = make([]float32, fsz)
	ev.FovDepthLogs = make([]float32, fsz)

	ev.MatMap = make(map[string]int, len(ev.Mats))
	for i, m := range ev.Mats {
		ev.MatMap[m] = i
	}
	ev.ActMap = make(map[string]int, len(ev.Acts))
	for i, m := range ev.Acts {
		ev.ActMap[m] = i
	}
	ev.USMap = make(map[string]int, len(ev.PosUSs)+len(ev.NegUSs))
	for i, m := range ev.PosUSs {
		ev.USMap[m] = i
	}
	for i, m := range ev.NegUSs {
		ev.USMap[m] = i
	}

	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Tick.Scale = env.Tick
	ev.Event.Scale = env.Event
	ev.Scene.Scale = env.Scene
	ev.Episode.Scale = env.Episode
}

func (ev *FWorld) Validate() error {
	if ev.Size.IsNil() {
		return fmt.Errorf("FWorld: %v has size == 0 -- need to Config", ev.Nm)
	}
	return nil
}

func (ev *FWorld) State(element string) etensor.Tensor {
	return ev.CurStates[element]
}

// String returns the current state as a string
func (ev *FWorld) String() string {
	return fmt.Sprintf("Evt_%d_Pos_%d_%d_Ang_%d_Act_%s", ev.Event.Cur, ev.PosI.X, ev.PosI.Y, ev.HeadDir, ev.Acts[ev.LastAct])
}

// Init is called to restart environment
func (ev *FWorld) Init(run int) {

	// note: could gen a new random world too..
	fnm := fmt.Sprintf("world_%d.tsv", mpi.WorldRank())
	ev.OpenWorld(gi.FileName(fnm))

	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Tick.Init()
	ev.Event.Init()
	ev.Scene.Init()
	ev.Episode.Init()

	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.Tick.Cur = -1
	ev.Event.Cur = -1

	ev.ShouldGate = false
	ev.JustGated = false
	ev.HasGated = false

	ev.PosI = ev.Size.DivScalar(2) // start in middle -- could be random..
	ev.PosI.Y -= 2                 // todo: special starting point!
	ev.PosI.X -= 6
	ev.PosF = ev.PosI.ToVec2()
	for i := 0; i < 4; i++ {
		ev.ProxMats[i] = 0
	}

	ev.HeadDir = 0
	ev.RotAng = 0

	ev.RefreshEvents = make(map[int]*WEvent)
	ev.AllEvents = make(map[int]*WEvent)
}

// InitPos sets initial position based on mpi node
func (ev *FWorld) InitPos(n int) {
	ypos := []int{8, ev.Size.Y / 2, ev.Size.Y - 8}
	nxpos := 12
	xpos := make([]int, nxpos)
	xpi := float32(ev.Size.X) / float32(nxpos+1)
	for i := 0; i < nxpos; i++ {
		xpos[i] = int(mat32.Round(float32(i+1) * xpi))
	}
	// fmt.Printf("%v\n", xpos)
	yi := n / nxpos
	xi := n % nxpos
	ev.PosI.X = xpos[xi]
	ev.PosI.Y = ypos[yi]
	ev.PosF = ev.PosI.ToVec2()
}

// SetWorld sets given mat at given point coord in world
func (ev *FWorld) SetWorld(p evec.Vec2i, mat int) {
	ev.World.Set([]int{p.Y, p.X}, mat)
}

// GetWorld returns mat at given point coord in world
func (ev *FWorld) GetWorld(p evec.Vec2i) int {
	return ev.World.Value([]int{p.Y, p.X})
}

////////////////////////////////////////////////////////////////////
// I/O

// SaveWorld saves the world to a tsv file with empty string for empty cells
func (ev *FWorld) SaveWorld(filename gi.FileName) error {
	fp, err := os.Create(string(filename))
	if err != nil {
		fmt.Println("Error creating file:", err)
		return err
	}
	defer fp.Close()
	bw := bufio.NewWriter(fp)
	for y := 0; y < ev.Size.Y; y++ {
		for x := 0; x < ev.Size.X; x++ {
			mat := ev.World.Value([]int{y, x})
			ms := ev.Mats[mat]
			if ms == "Empty" {
				ms = ""
			}
			bw.WriteString(ms + "\t")
		}
		bw.WriteString("\n")
	}
	bw.Flush()
	return nil
}

// OpenWorld loads the world from a tsv file with empty string for empty cells
func (ev *FWorld) OpenWorld(filename gi.FileName) error {
	fp, err := os.Open(string(filename))
	if err != nil {
		fmt.Println("Error opening file:", err)
		return err
	}
	defer fp.Close()
	ev.World.SetZeros()
	scan := bufio.NewScanner(fp)
	for y := 0; y < ev.Size.Y; y++ {
		if !scan.Scan() {
			break
		}
		ln := scan.Bytes()
		sz := len(ln)
		if sz == 0 {
			break
		}
		sp := bytes.Split(ln, []byte("\t"))
		sz = ints.MinInt(ev.Size.X, len(sp)-1)
		for x := 0; x < ev.Size.X; x++ {
			ms := string(sp[x])
			if ms == "" {
				continue
			}
			mi, ok := ev.MatMap[ms]
			if !ok {
				fmt.Printf("Mat not found: %s\n", ms)
			} else {
				ev.World.Set([]int{y, x}, mi)
			}
		}
	}
	return nil
}

// SavePats saves the patterns
func (ev *FWorld) SavePats(filename gi.FileName) error {
	jenc, _ := json.MarshalIndent(ev.Pats, "", " ")
	return ioutil.WriteFile(string(filename), jenc, 0644)
}

// OpenPats opens the patterns
func (ev *FWorld) OpenPats(filename gi.FileName) error {
	fp, err := os.Open(string(filename))
	if err != nil {
		fmt.Println("Error opening file:", err)
		return err
	}
	defer fp.Close()
	b, err := ioutil.ReadAll(fp)
	err = json.Unmarshal(b, &ev.Pats)
	if err != nil {
		fmt.Println(err)
	}
	return err
}

// AngMod returns angle modulo within 360 degrees
func AngMod(ang int) int {
	if ang < 0 {
		ang += 360
	} else if ang > 360 {
		ang -= 360
	}
	return ang
}

// AngVec returns the incremental vector to use for given angle, in deg
// such that the largest value is 1.
func AngVec(ang int) mat32.Vec2 {
	a := mat32.DegToRad(float32(AngMod(ang)))
	v := mat32.Vec2{mat32.Cos(a), mat32.Sin(a)}
	return NormVecLine(v)
}

// NormVec normalize vector for drawing a line
func NormVecLine(v mat32.Vec2) mat32.Vec2 {
	av := v.Abs()
	if av.X > av.Y {
		v = v.DivScalar(av.X)
	} else {
		v = v.DivScalar(av.Y)
	}
	return v
}

// NextVecPoint returns the next grid point along vector,
// from given current floating and grid points.  v is normalized
// such that the largest value is 1.
func NextVecPoint(cp, v mat32.Vec2) (mat32.Vec2, evec.Vec2i) {
	n := cp.Add(v)
	g := evec.NewVec2iFmVec2Round(n)
	return n, g
}

////////////////////////////////////////////////////////////////////
// Vision

// ScanDepth does simple ray-tracing to find depth and material along each angle vector
func (ev *FWorld) ScanDepth() {
	nmat := len(ev.Mats)
	_ = nmat
	idx := 0
	hang := ev.FOV / 2
	maxld := mat32.Log(1 + mat32.Sqrt(float32(ev.Size.X*ev.Size.X+ev.Size.Y*ev.Size.Y)))
	for ang := hang; ang >= -hang; ang -= ev.VisAngInc {
		v := AngVec(ang + ev.HeadDir)
		op := ev.PosF
		cp := op
		gp := evec.Vec2i{}
		depth := float32(-1)
		vmat := 0 // first non-empty visible material
		for {
			cp, gp = NextVecPoint(cp, v)
			if gp.X < 0 || gp.X >= ev.Size.X {
				break
			}
			if gp.Y < 0 || gp.Y >= ev.Size.Y {
				break
			}
			mat := ev.GetWorld(gp)
			if mat > 0 && mat <= ev.BarrierIdx {
				vmat = mat
				depth = cp.DistTo(op)
				break
			}
			if ev.ShowRays {
				ev.SetWorld(gp, nmat+idx*2) // visualization
			}
		}
		ev.Depths[idx] = depth
		ev.ViewMats[idx] = vmat
		if depth > 0 {
			ev.DepthLogs[idx] = mat32.Log(1+depth) / maxld
		} else {
			ev.DepthLogs[idx] = 1
		}
		idx++
	}
}

// ScanFovea does simple ray-tracing to find depth and material for fovea
func (ev *FWorld) ScanFovea() {
	nmat := len(ev.Mats)
	idx := 0
	maxld := mat32.Log(1 + mat32.Sqrt(float32(ev.Size.X*ev.Size.X+ev.Size.Y*ev.Size.Y)))
	for fi := -ev.FoveaSize; fi <= ev.FoveaSize; fi++ {
		ang := -fi * ev.FoveaAngInc
		v := AngVec(ang + ev.HeadDir)
		op := ev.PosF
		cp := op
		gp := evec.Vec2i{}
		depth := float32(-1)
		vmat := 0 // first non-empty visible material
		for {
			cp, gp = NextVecPoint(cp, v)
			if gp.X < 0 || gp.X >= ev.Size.X {
				break
			}
			if gp.Y < 0 || gp.Y >= ev.Size.Y {
				break
			}
			mat := ev.GetWorld(gp)
			if mat > 0 && mat < nmat {
				vmat = mat
				depth = cp.DistTo(op)
				break
			}
			if ev.ShowFovRays {
				ev.SetWorld(gp, nmat+idx*2) // visualization
			}
		}
		ev.FovDepths[idx] = depth
		ev.FovMats[idx] = vmat
		if depth > 0 {
			ev.FovDepthLogs[idx] = mat32.Log(1+depth) / maxld
		} else {
			ev.FovDepthLogs[idx] = 1
		}
		idx++
	}
}

// ScanProx scan the proximal space around the agent
func (ev *FWorld) ScanProx() {
	angs := []int{0, -90, 90, 180}
	for i := 0; i < 4; i++ {
		v := AngVec(ev.HeadDir + angs[i])
		_, gp := NextVecPoint(ev.PosF, v)
		ev.ProxMats[i] = ev.GetWorld(gp)
		ev.ProxPos[i] = gp
	}
}

////////////////////////////////////////////////////////////////////
// Actions

// WEvent records an event
type WEvent struct {
	Tick   int        `desc:"tick when event happened"`
	PosI   evec.Vec2i `desc:"discrete integer grid position where event happened"`
	PosF   mat32.Vec2 `desc:"floating point grid position where event happened"`
	Angle  int        `desc:"angle pointing when event happened"`
	Act    int        `desc:"action that took place"`
	Mat    int        `desc:"material that was involved (front fovea mat)"`
	MatPos evec.Vec2i `desc:"position of material involved in event"`
}

// NewEvent returns new event with current state and given act, mat
func (ev *FWorld) NewEvent(act, mat int, matpos evec.Vec2i) *WEvent {
	return &WEvent{Tick: ev.Tick.Cur, PosI: ev.PosI, PosF: ev.PosF, Angle: ev.HeadDir, Act: act, Mat: mat, MatPos: matpos}
}

// AddNewEventRefresh adds event to RefreshEvents (a consumable was consumed).
// always adds to AllEvents
func (ev *FWorld) AddNewEventRefresh(wev *WEvent) {
	ev.RefreshEvents[wev.Tick] = wev
	ev.AllEvents[wev.Tick] = wev
}

// RefreshWorld refreshes consumables
func (ev *FWorld) RefreshWorld() {
	ct := ev.Tick.Cur
	refresh := int(ev.Params["EnvRefresh"])
	for t, wev := range ev.RefreshEvents {
		setmat := 0
		if t+refresh < ct {
			setmat = wev.Mat
			ev.SetWorld(wev.MatPos, setmat)
			delete(ev.RefreshEvents, t)
		}
	}
}

// DecodeAct decodes action from given tensor of activation states
// Forward is only selected if it is 2x larger than other options
func (ev *FWorld) DecodeAct(vt *etensor.Float32) int {
	cnm := ""
	var dst, fdst float32
	for nm, pat := range ev.ActPats {
		d := metric.Correlation32(vt.Values, pat.Values)
		if nm == "Forward" {
			fdst = d
		} else {
			if cnm == "" || d > dst {
				cnm = nm
				dst = d
			}
		}
	}
	if fdst > ev.FwdMargin*dst { // only if x bigger
		cnm = "Forward"
	}
	act, ok := ev.ActMap[cnm]
	if !ok {
		act = rand.Intn(len(ev.Acts))
	}
	return act
}

// TakeAct takes the action, updates state
func (ev *FWorld) TakeAct(act int) {
	as := ""
	if act >= len(ev.Acts) || act < 0 {
		as = "Stay"
	} else {
		as = ev.Acts[act]
	}

	ev.RotAng = 0

	ev.ClearUS("PosUSs")
	ev.ClearUS("NegUSs")

	if as == "None" {
		return
	}

	nmat := len(ev.Mats)
	proxMat := ints.MinInt(ev.ProxMats[0], nmat)
	behmat := ev.ProxMats[3]  // behind
	front := ev.Mats[proxMat] // state in front

	eff := float32(0)

	mve := ev.Params["MoveEffort"]
	rote := ev.Params["RotEffort"]
	bumpp := ev.Params["BumpPain"]

	switch as {
	case "Stay":
	case "Left":
		ev.RotAng = ev.MotAngInc
		ev.HeadDir = AngMod(ev.HeadDir + ev.RotAng)
		eff += rote
	case "Right":
		ev.RotAng = -ev.MotAngInc
		ev.HeadDir = AngMod(ev.HeadDir + ev.RotAng)
		eff += rote
	case "Forward":
		eff += mve
		if proxMat > 0 && proxMat <= ev.BarrierIdx {
			ev.SetUS("NegUSs", "Bump", bumpp)
		} else {
			ev.PosF, ev.PosI = NextVecPoint(ev.PosF, AngVec(ev.HeadDir))
		}
	case "Backward":
		eff += mve
		if behmat > 0 && behmat <= ev.BarrierIdx {
			ev.SetUS("NegUSs", "Bump", bumpp)
		} else {
			ev.PosF, ev.PosI = NextVecPoint(ev.PosF, AngVec(AngMod(ev.HeadDir+180)))
		}
	case "Consume":
		if proxMat >= ev.MatsUSStart {
			ev.SetUS("PosUSs", front, 1)
			ev.AddNewEventRefresh(ev.NewEvent(act, proxMat, ev.ProxPos[0]))
			ev.SetWorld(ev.ProxPos[0], proxMat+ev.NDrives)
			ev.Event.Set(0)
			ev.Scene.Incr()
		}
	}
	ev.LastEffort = eff
	ev.ScanDepth()
	ev.ScanFovea()
	ev.ScanProx()

	ev.RenderState()
}

// RenderView renders the current view state to NextStates tensor input states
func (ev *FWorld) RenderView() {
	dv := ev.NextStates["Depth"]
	for i := 0; i < ev.NFOVRays; i++ {
		sv := dv.SubSpace([]int{0, i}).(*etensor.Float32)
		ev.DepthCode.Encode(&sv.Values, ev.DepthLogs[i], ev.DepthSize, popcode.Set)
	}

	fsz := 1 + 2*ev.FoveaSize
	fd := ev.NextStates["FovDepth"]
	fv := ev.NextStates["Fovea"]
	for i := 0; i < fsz; i++ {
		sv := fd.SubSpace([]int{0, i}).(*etensor.Float32)
		ev.DepthCode.Encode(&sv.Values, ev.FovDepthLogs[i], ev.DepthSize, popcode.Set)
		fm := ev.FovMats[i]
		if fm < len(ev.Mats) {
			sv := fv.SubSpace([]int{0, i}).(*etensor.Float32)
			ms := ev.Mats[fm]
			mp, ok := ev.Pats[ms]
			if ok {
				sv.CopyFrom(mp)
			}
		}
	}
}

// RenderProxSoma renders proximal soma state
func (ev *FWorld) RenderProxSoma() {
	ps := ev.NextStates["ProxSoma"]
	ps.SetZeros()
	for i := 0; i < 4; i++ {
		if ev.ProxMats[i] != 0 {
			ps.Set([]int{0, i, 0, 0}, 1) // on
		} else {
			ps.Set([]int{0, i, 1, 0}, 1) // off
		}
	}
}

// RenderVestibular renders vestibular state
func (ev *FWorld) RenderVestibular() {
	vs := ev.NextStates["HeadDir"]
	nv := (float32(ev.HeadDir) / 360.0)
	ev.AngCode.Encode(&vs.Values, nv, ev.PopSize)
}

// RenderAction renders action pattern
func (ev *FWorld) RenderAction() {
	av := ev.NextStates["Action"]
	if ev.LastAct < len(ev.Acts) {
		as := ev.Acts[ev.LastAct]
		ap, ok := ev.Pats[as]
		if ok {
			av.CopyFrom(ap)
		}
	}
}

// ClearUS resets US values to 0
func (ev *FWorld) ClearUS(ustype string) {
	us := ev.NextStates[ustype]
	us.SetZeros()
}

// SetUS sets given us value
func (ev *FWorld) SetUS(ustype, us string, val float32) {
	uss := ev.NextStates[ustype]
	ix := ev.USMap[us]
	uss.Set1D(ix, val)
}

// RenderState renders the current state into NextState vars
func (ev *FWorld) RenderState() {
	ev.RenderView()
	ev.RenderProxSoma()
	ev.RenderVestibular()
	ev.RenderAction()
}

// CopyNextToCur copy next state to current state
func (ev *FWorld) CopyNextToCur() {
	for k, ns := range ev.NextStates {
		cs, ok := ev.CurStates[k]
		if !ok {
			cs = ns.Clone().(*etensor.Float32)
			ev.CurStates[k] = cs
		} else {
			cs.CopyFrom(ns)
		}
	}
}

// Step is called to advance the environment state
func (ev *FWorld) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.RenderAction()
	ev.CopyNextToCur()
	ev.Tick.Incr()
	ev.Event.Incr()
	ev.RefreshWorld()
	if ev.Trial.Incr() { // true if wraps around Max back to 0
		ev.Epoch.Incr()
	}
	return true
}

func (ev *FWorld) Action(action string, nop etensor.Tensor) {
	a, ok := ev.ActMap[action]
	if !ok {
		fmt.Printf("Action not recognized: %s\n", action)
		return
	}
	ev.LastAct = a
	ev.TakeAct(ev.LastAct)
}

func (ev *FWorld) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	case env.Tick:
		return ev.Tick.Query()
	case env.Event:
		return ev.Event.Query()
	case env.Scene:
		return ev.Scene.Query()
	case env.Episode:
		return ev.Episode.Query()
	}
	return -1, -1, false
}

////////////////////////////////////////////////////////////////////
// Render world

// WorldLineHoriz draw horizontal line
func (ev *FWorld) WorldLineHoriz(st, ed evec.Vec2i, mat int) {
	sx := ints.MinInt(st.X, ed.X)
	ex := ints.MaxInt(st.X, ed.X)
	for x := sx; x <= ex; x++ {
		ev.World.Set([]int{st.Y, x}, mat)
	}
}

// WorldLineVert draw vertical line
func (ev *FWorld) WorldLineVert(st, ed evec.Vec2i, mat int) {
	sy := ints.MinInt(st.Y, ed.Y)
	ey := ints.MaxInt(st.Y, ed.Y)
	for y := sy; y <= ey; y++ {
		ev.World.Set([]int{y, st.X}, mat)
	}
}

// WorldLine draw line in world with given mat
func (ev *FWorld) WorldLine(st, ed evec.Vec2i, mat int) {
	di := ed.Sub(st)

	if di.X == 0 {
		ev.WorldLineVert(st, ed, mat)
		return
	}
	if di.Y == 0 {
		ev.WorldLineHoriz(st, ed, mat)
		return
	}

	dv := di.ToVec2()
	dst := dv.Length()
	v := NormVecLine(dv)
	op := st.ToVec2()
	cp := op
	gp := evec.Vec2i{}
	for {
		cp, gp = NextVecPoint(cp, v)
		ev.SetWorld(gp, mat)
		d := cp.DistTo(op) // not very efficient, but works.
		if d >= dst {
			break
		}
	}
}

// WorldRandom distributes n of given material in random locations
func (ev *FWorld) WorldRandom(n, mat int) {
	cnt := 0
	for cnt < n {
		px := rand.Intn(ev.Size.X)
		py := rand.Intn(ev.Size.Y)
		ix := []int{py, px}
		cm := ev.World.Value(ix)
		if cm == 0 {
			ev.World.Set(ix, mat)
			cnt++
		}
	}
}

// WorldRect draw rectangle in world with given mat
func (ev *FWorld) WorldRect(st, ed evec.Vec2i, mat int) {
	ev.WorldLineHoriz(st, evec.Vec2i{ed.X, st.Y}, mat)
	ev.WorldLineHoriz(evec.Vec2i{st.X, ed.Y}, evec.Vec2i{ed.X, ed.Y}, mat)
	ev.WorldLineVert(st, evec.Vec2i{st.X, ed.Y}, mat)
	ev.WorldLineVert(evec.Vec2i{ed.X, st.Y}, evec.Vec2i{ed.X, ed.Y}, mat)
}

// GenWorld generates a world -- edit to create in way desired
func (ev *FWorld) GenWorld() {
	wall := ev.MatMap["Wall"]
	ev.World.SetZeros()
	// always start with a wall around the entire world -- no seeing the turtles..
	ev.WorldRect(evec.Vec2i{0, 0}, evec.Vec2i{ev.Size.X - 1, ev.Size.Y - 1}, wall)

	// additional obstacles:
	// ev.WorldRect(evec.Vec2i{20, 20}, evec.Vec2i{40, 40}, wall)
	// ev.WorldRect(evec.Vec2i{60, 60}, evec.Vec2i{80, 80}, wall)
	// ev.WorldLine(evec.Vec2i{60, 20}, evec.Vec2i{80, 40}, wall) // double-thick lines = no leak
	// ev.WorldLine(evec.Vec2i{60, 19}, evec.Vec2i{80, 39}, wall)

	// don't put anything in center starting point
	ctr := ev.Size.DivScalar(2)
	ev.SetWorld(ctr, wall)

	nper := 40 / ev.NDrives // was 100 for 100x100 large
	for i := 0; i < ev.NDrives; i++ {
		ev.WorldRandom(nper, ev.MatsUSStart+i)
	}

	// clear center
	ev.SetWorld(ctr, 0)
}

////////////////////////////////////////////////////////////////////
// Subcortex / Instinct

// InstinctTrace prints trace of act gen if enabled
func (ev *FWorld) InstinctTrace(desc string, act int) {
	if !ev.TraceInstinct {
		return
	}
	ev.TraceInst = fmt.Sprintf("%s: act: %s", desc, ev.Acts[act])
	fmt.Println(ev.TraceInst)
}

// ReadFovea returns the contents of the fovea
func (ev *FWorld) ReadFovea() (foodDepth, foodWeight, waterDepth, waterWeight, foveaDepth float32, foveaMat, foveaNonWall int, foveaMatName string) {
	fsz := 1 + 2*ev.FoveaSize
	food := ev.MatMap["Food"]
	water := ev.MatMap["Water"]

	foodWeight = 0
	waterWeight = 0
	foodDepth = 100000
	waterDepth = 100000
	foveaDepth = 100000
	foveaNonWall = 0
	for i := 0; i < fsz; i++ {
		mat := ev.FovMats[i]
		switch {
		case mat == water:
			waterWeight += 1 - ev.FovDepthLogs[i] // more weight if closer
			waterDepth = mat32.Min(waterDepth, ev.FovDepths[i])
		case mat == food:
			foodWeight += 1 - ev.FovDepthLogs[i] // more weight if closer
			foodDepth = mat32.Min(foodDepth, ev.FovDepths[i])
		case mat <= ev.BarrierIdx:
		default:
			foveaNonWall = mat
		}
		foveaDepth = mat32.Min(foveaDepth, ev.FovDepths[i])
	}
	// foodWeight *= 1 - ev.InterStates["Energy"] // weight by need
	// waterWeight *= 1 - ev.InterStates["Hydra"]

	foveaMat = ev.FovMats[ev.FoveaSize]
	foveaMatName = ev.Mats[foveaMat]
	return
}

// ReadFullField reads the full wide field, returning proximity of
// barrier in left and right visual fields
func (ev *FWorld) ReadFullField() (leftClose, rightClose float32) {
	minLeftDepth := float32(1.0)
	minRightDepth := float32(1.0)
	avgLeftDepth := float32(0.0)
	avgRightDepth := float32(0.0)
	halfAng := ev.NFOVRays / 2
	for i := 0; i < ev.NFOVRays; i++ {
		dp := float32(ev.DepthLogs[i])
		if i < halfAng-1 {
			minLeftDepth = mat32.Min(minLeftDepth, dp)
			avgLeftDepth += dp
		} else if i > halfAng+1 {
			minRightDepth = mat32.Min(minRightDepth, dp)
			avgRightDepth += dp
		}
	}
	leftClose = 1 - minLeftDepth
	rightClose = 1 - minRightDepth
	if mat32.Abs(minLeftDepth-minRightDepth) < 0.1 { // if close tie on min
		leftClose = 1 - (avgLeftDepth / float32(halfAng-1)) // go with average
		rightClose = 1 - (avgRightDepth / float32(halfAng-1))
	}
	return
}

// InstinctAct generates an action for current situation based on simple
// coded heuristics -- i.e., what subcortical evolutionary instincts provide.
// Also returns the urgency score as a probability -- if urgency is 1
// then the generated action should definitely be used.  The default is 0,
// which is the baseline.
func (ev *FWorld) InstinctAct(justGated, hasGated bool) (int, float32) {
	ev.JustGated = justGated
	ev.HasGated = hasGated

	wall := ev.MatMap["Wall"]
	fwd := ev.ActMap["Forward"]
	left := ev.ActMap["Left"]
	right := ev.ActMap["Right"]
	consume := ev.ActMap["Consume"]

	nmat := len(ev.Mats)
	proxMat := ints.MinInt(ev.ProxMats[0], nmat)

	foodDepth, foodWeight, waterDepth, waterWeight, foveaDepth, foveaMat, foveaNonWall, foveaMatName := ev.ReadFovea()
	_ = foveaMat

	leftClose, rightClose := ev.ReadFullField()

	smaxpow := float32(10.0)
	rlp := float32(.5)
	if leftClose+rightClose > 0 {
		rpow := mat32.Exp(rightClose * smaxpow)
		lpow := mat32.Exp(leftClose * smaxpow)
		rlp = lpow / (rpow + lpow)
	}
	rlact := left // right or left
	if erand.BoolP(float64(rlp), -1) {
		rlact = right
	}
	// fmt.Printf("rlp: %.3g  leftClose: %.3g  rightClose: %.3g  act: %s\n", rlp, leftClose, rightClose, ev.Acts[rlact])
	rlps := fmt.Sprintf("%.3g", rlp)

	lastact := ev.LastAct
	frnd := rand.Float32()

	farDist := float32(10)
	farTurnP := float32(0.2)
	rndExpSame := float32(0.33)
	rndExpTurn := float32(0.33)

	ev.ShouldGate = false

	urgency := float32(0)
	act := ev.ActMap["Forward"] // default
	switch {
	case proxMat == wall:
		if lastact == left || lastact == right {
			act = lastact // keep going
			ev.InstinctTrace("at wall, keep turning", act)
		} else {
			act = rlact
			ev.InstinctTrace(fmt.Sprintf("at wall, rlp: %s, turn", rlps), act)
		}
		urgency = ev.WallUrgency
	case proxMat >= ev.MatsUSStart && proxMat < ev.MatsUSStart+ev.NDrives:
		ev.ShouldGate = true
		act = consume
		ev.InstinctTrace("at US, consume", act)
		urgency = ev.EatUrgency
	case ev.HasGated:
		act = fwd
		ev.InstinctTrace("has gated", act)
	case foodWeight > waterWeight: // food more than water
		wts := fmt.Sprintf("foodWeight: %g > waterWeight: %g, dist: %g", foodWeight, waterWeight, foodDepth)
		if foodDepth > farDist { // far away
			urgency = 0
			if frnd < farTurnP {
				act = rlact
				ev.InstinctTrace(fmt.Sprintf("far food in view (%s), explore, rlp: %s, turn", wts, rlps), act)
			} else {
				ev.InstinctTrace("far food in view "+wts, act)
			}
		} else {
			urgency = ev.CloseUrgency
			ev.InstinctTrace("close food in view "+wts, act)
		}
	case waterWeight > foodWeight: // water more than food
		wts := fmt.Sprintf("waterWeight: %g > foodWeight: %g, dist: %g", waterWeight, foodWeight, waterDepth)
		if waterDepth > farDist { // far away
			urgency = 0
			if frnd < farTurnP {
				act = rlact
				ev.InstinctTrace(fmt.Sprintf("far water in view (%s), explore, rlp: %s, turn", wts, rlps), act)
			} else {
				ev.InstinctTrace("far water in view "+wts, act)
			}
		} else {
			urgency = ev.CloseUrgency
			ev.InstinctTrace("close water in view "+wts, act)
		}
	case foveaDepth < 4 && foveaNonWall == 0: // close to wall
		urgency = ev.CloseUrgency
		if lastact == left || lastact == right {
			act = lastact // keep going
			ev.InstinctTrace("close to: "+foveaMatName+" keep turning", act)
		} else {
			act = rlact
			ev.InstinctTrace(fmt.Sprintf("close to: %s rlp: %s, turn", foveaMatName, rlps), act)
		}
	default: // random explore -- nothing obvious
		urgency = 0
		switch {
		case frnd < rndExpSame && lastact < consume:
			act = lastact // continue
			ev.InstinctTrace("looking at: "+foveaMatName+" repeat last act", act)
		case frnd < rndExpSame+rndExpTurn:
			act = rlact
			ev.InstinctTrace("looking at: "+foveaMatName+" turn", act)
		default:
			ev.InstinctTrace("looking at: "+foveaMatName+" go", act)
		}
	}

	ev.Urgency = urgency
	return act, urgency
}

// Compile-time check that implements Env interface
var _ env.Env = (*FWorld)(nil)

var FWorldProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"OpenWorld", ki.Props{
			"label": "Open World...",
			"icon":  "file-open",
			"desc":  "Open World from tsv file",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".tsv",
				}},
			},
		}},
		{"SaveWorld", ki.Props{
			"label": "Save World...",
			"icon":  "file-save",
			"desc":  "Save World to tsv file",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".tsv",
				}},
			},
		}},
		{"OpenPats", ki.Props{
			"label": "Open Pats...",
			"icon":  "file-open",
			"desc":  "Open pats from json file",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".json",
				}},
			},
		}},
		{"SavePats", ki.Props{
			"label": "Save Pats...",
			"icon":  "file-save",
			"desc":  "Save pats to json file",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".json",
				}},
			},
		}},
	},
}
