// Copyright (c) 2019, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package navenv

import (
	"image"
	"log"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/popcode"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/tsragg"
	"github.com/emer/eve/eve"
	"github.com/emer/eve/evev"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gi3d"
	"github.com/goki/gi/giv"
	"github.com/goki/gi/mat32"
	"github.com/goki/gi/oswin"
	"github.com/goki/gi/oswin/gpu"
	"github.com/goki/gi/units"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// Actions is a list of available actions for model
type Actions int

//go:generate stringer -type=Actions

var KiT_Actions = kit.Enums.AddEnum(ActionsN, false, nil)

// The actions avail
const (
	// forward implies rot body to head
	StepForward Actions = iota
	RotBodyLeft
	RotBodyRight
	RotHeadLeft
	RotHeadRight
	// StepBackward
	// RotBodyToHead
	// RotHeadToBody
	ActionsN
)

// ActionsCode are code letters for the actions
var ActionsCode = []string{"F", "L", "R", "l", "r"}

// Env manages the navigation environment
type Env struct {
	Nm               string           `desc:"name of this environment"`
	Dsc              string           `desc:"description of this environment"`
	Run              env.Ctr          `view:"inline" desc:"current run of model as provided during Init"`
	Epoch            env.Ctr          `view:"inline" desc:"number of times through arbitrary number of Events"`
	Event            env.Ctr          `view:"inline" desc:"current ordinal item in Table -- if Sequential then = row number in table, otherwise is index in Order list that then gives row number in Table"`
	EmerHt           float32          `desc:"height of emer"`
	MoveStep         float32          `desc:"how far to move every step"`
	RotStep          float32          `desc:"how far to rotate every step"`
	PosRes           int              `desc:"number of grid points for encoding position in position state"`
	PosPop           popcode.TwoD     `desc:"2D population code parameters for encoding position, in normalized position units (0-1)"`
	AngRes           int              `desc:"number of units for encoding angles and related velocities etc"`
	AngPop           popcode.OneD     `desc:"1D population code parameters for encoding angles, in normalized position units (0-1)"`
	VelPop           popcode.OneD     `desc:"1D population code parameters for encoding angles, in normalized units (-1..1)"`
	ActRes           int              `desc:"number of units for encoding actions"`
	ActVar           float64          `desc:"gaussian variance of noise in action for *selected* actions"`
	ActNoise         float64          `desc:"gaussian variance of noise in action for unselected actions"`
	ActPop           popcode.OneD     `desc:"1D population code parameters for encoding actions, in normalized units (0..1)"`
	Room             RoomParams       `desc:"parameters for room"`
	Camera           evev.Camera      `desc:"offscreen render camera settings"`
	Policy           Policy           `view:"inline" desc:"current policy for actions"`
	DepthMap         giv.ColorMapName `desc:"color map to use for rendering depth map"`
	CurImage         image.Image      `desc:"current first-person image"`
	RawDepth         etensor.Float32  `desc:"raw depth map X x Y same size as camera"`
	CurDepth         etensor.Float32  `desc:"current normalized depth map X x Y same size as camera"`
	CurPos           mat32.Vec2       `desc:"current normalized position"`
	CurPosMap        etensor.Float32  `desc:"current normalized position map X x Y pos res"`
	HeadDir          env.CurPrvF32    `desc:"normalized head direction (0 = 'north' = Z)"`
	CurHeadDirMap    etensor.Float32  `desc:"current normalized head direction map"`
	CurHeadVelMap    etensor.Float32  `desc:"current normalized head direction map"`
	NeckAng          env.CurPrvF32    `desc:"normalized neck angle relative to body"`
	CurNeckAngMap    etensor.Float32  `desc:"current normalized neck angle map"`
	CurNeckAngVelMap etensor.Float32  `desc:"current normalized neck angle veolicity map"`
	CurSomaMap       etensor.Float32  `desc:"all of the head dir, angle, neck info in one place"`
	CurAct           Actions          `desc:"current action selected"`
	PrvAct           Actions          `desc:"previous action selected"`
	ExtAct           Actions          `desc:"current externally-supplied action"`
	CurActMag        float32          `desc:"magnitude of current action -- how strong"`
	CurActMap        etensor.Float32  `desc:"action as a pop-coded map, returned as state"`
	World            *eve.Group       `view:"-" desc:"world"`
	View             *evev.View       `view:"-" desc:"view of world"`
	Emer             *eve.Group       `view:"-" desc:"emer group"`
	Head             *eve.Group       `view:"-" desc:"Head of emer"`
	EyeR             eve.Body         `view:"-" desc:"Right eye of emer"`
	Win              *gi.Window       `view:"-" desc:"gui window -- can be nil"`
	SnapImg          *gi.Bitmap       `view:"-" desc:"snapshot bitmap view"`
	DepthImg         *gi.Bitmap       `view:"-" desc:"depth map bitmap view"`
	Frame            gpu.Framebuffer  `view:"-" desc:"offscreen render buffer"`
}

func (ev *Env) Name() string { return ev.Nm }
func (ev *Env) Desc() string { return ev.Dsc }

func (ev *Env) Validate() error {
	return nil
}

func (ev *Env) Defaults() {
	ev.Room.Defaults()
	ev.Policy.Defaults()
	ev.EmerHt = 1
	ev.MoveStep = ev.EmerHt * .4
	ev.RotStep = 15
	ev.PosRes = 16
	ev.AngRes = 16
	ev.ActRes = 16
	ev.ActVar = 0.5
	ev.ActNoise = 0.1 // gaussian noise
	ev.PosPop.Defaults()
	ev.PosPop.SetRange(0, 1, 0.05)
	ev.AngPop.Defaults()
	ev.AngPop.SetRange(-0.2, 1.2, 0.1)
	ev.VelPop.Defaults()
	ev.VelPop.SetRange(-1.2, 1.2, 0.1)
	ev.ActPop.Defaults()
	ev.ActPop.SetRange(-0.5, 1.5, 0.15)
	ev.DepthMap = giv.ColorMapName("ColdHot")
	ev.Camera.Defaults()
	ev.Camera.Size.X = 16
	ev.Camera.Size.Y = 8
	ev.Camera.FOV = 90
	ev.Camera.MaxD = 7
}

func (ev *Env) Init(run int) {
	if ev.World == nil {
		ev.MakeWorld()
	} else {
		ev.InitWorld()
		ev.UpdateWorld()
	}
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Event.Scale = env.Event
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Event.Init()
	ev.Run.Cur = run
	ev.CurActMag = 1
	ev.Event.Cur = -1 // init state -- key so that first Step() = 0
	ev.RawDepth.SetShape([]int{ev.Camera.Size.Y, ev.Camera.Size.X}, nil, []string{"Y", "X"})
	ev.CurDepth.SetShape([]int{ev.Camera.Size.Y, ev.Camera.Size.X}, nil, []string{"Y", "X"})
	ev.CurPosMap.SetShape([]int{ev.PosRes, ev.PosRes}, nil, []string{"Y", "X"})
	ev.CurHeadDirMap.SetShape([]int{ev.AngRes}, nil, nil)
	ev.CurHeadVelMap.SetShape([]int{ev.AngRes}, nil, nil)
	ev.CurNeckAngMap.SetShape([]int{ev.AngRes}, nil, nil)
	ev.CurNeckAngVelMap.SetShape([]int{ev.AngRes}, nil, nil)
	ev.CurSomaMap.SetShape([]int{4, 1, 1, ev.AngRes}, nil, []string{"Type", "1", "1", "Value"})
	ev.CurActMap.SetShape([]int{int(ActionsN), 1, 1, ev.ActRes}, nil, []string{"Action", "1", "1", "Mag"})
}

func (ev *Env) Step() bool {
	mind := float32(tsragg.Min(&ev.CurDepth))
	avgd := float32(tsragg.Mean(&ev.CurDepth))
	if mind == 1 && avgd == 1 { // actually
		mind = 0
		avgd = 0
	} else if mind == 0 && avgd == 0 {
		mind = 0.5
		avgd = 0.5
	}
	ev.CurAct = ev.Policy.Act(mind, avgd, ev.CurPos, ev.ExtAct)
	if ev.CurAct != ev.ExtAct {
		ev.CurActMag = float32((1 - ev.ActNoise) + erand.Gauss(ev.ActVar, -1))
		ev.CurActMag = mat32.Clamp(ev.CurActMag, 0, 1)
	}
	ev.TakeAction(ev.CurAct)
	ev.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ev.Event.Incr() { // if true, hit max, reset to 0
		ev.Epoch.Incr()
	}
	return true
}

func (ev *Env) States() env.Elements {
	els := env.Elements{
		{"Depth", []int{ev.Camera.Size.Y, ev.Camera.Size.X}, []string{"Y", "X"}},
		{"PosMap", []int{ev.PosRes, ev.PosRes}, []string{"Y", "X"}},
		{"ActMap", []int{int(ActionsN)}, []string{"ActionsN"}},
		{"HeadDir", []int{int(ev.AngRes)}, []string{"N"}},
		{"HeadVel", []int{int(ev.AngRes)}, []string{"N"}},
		{"NeckAng", []int{int(ev.AngRes)}, []string{"N"}},
		{"NeckAngVel", []int{int(ev.AngRes)}, []string{"N"}},
		{"SomaMap", []int{4, 1, 1, int(ev.AngRes)}, []string{"Type", "1", "1", "Value"}}, // all-in-one, 4D
	}
	return els
}

func (ev *Env) State(element string) etensor.Tensor {
	switch element {
	case "Depth":
		return &ev.CurDepth
	case "PosMap":
		return &ev.CurPosMap
	case "ActMap":
		return &ev.CurActMap
	case "HeadDir":
		return &ev.CurHeadDirMap
	case "HeadVel":
		return &ev.CurHeadVelMap
	case "NeckAng":
		return &ev.CurNeckAngMap
	case "NeckAngVel":
		return &ev.CurNeckAngVelMap
	case "SomaMap":
		return &ev.CurSomaMap
	}
	return nil
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
	if input.DataType() == etensor.STRING {
		astr := input.StringVal1D(0)
		ev.CurAct.FromString(astr)
	} else {
		ev.CurAct = Actions(input.FloatVal1D(0))
	}
}

// SetAction is easier non-standard interface just for this -- mag is 0..1 normalized magnitude of action
func (ev *Env) SetAction(act Actions, mag float32) {
	ev.PrvAct = ev.CurAct
	ev.ExtAct = act
	ev.CurAct = act
	ev.CurActMag = mag
}

// MakeWorld constructs a new virtual physics world
func (ev *Env) MakeWorld() {
	ev.World = &eve.Group{}
	ev.World.InitName(ev.World, "RoomWorld")

	ev.Room.MakeRoom(ev.World, "room1")
	ev.Emer = MakeEmer(ev.World, ev.EmerHt)
	ev.Head = ev.Emer.ChildByName("head", 1).(*eve.Group)
	ev.EyeR = ev.Head.ChildByName("eye-r", 2).(eve.Body)

	ev.World.InitWorld()
}

// InitWorld does init on world and re-syncs
func (ev *Env) InitWorld() {
	ev.World.InitWorld()
	if ev.View != nil {
		ev.View.Sync()
	}
}

// ReMakeWorld rebuilds the world and re-syncs with gui
func (ev *Env) ReMakeWorld() {
	ev.MakeWorld()
	if ev.View != nil {
		ev.View.World = ev.World
		ev.View.Sync()
		ev.UpdateView()
	}
}

// MakeView makes the view
func (ev *Env) MakeView(sc *gi3d.Scene) {
	wgp := gi3d.AddNewGroup(sc, sc, "world")
	ev.View = evev.NewView(ev.World, sc, wgp)
	ev.View.Sync()
}

// StepForward moves Emer forward in current facing direction one step, and updates
func (ev *Env) StepForward() {
	ev.RotBodyToHead()
	ev.Emer.Rel.MoveOnAxis(0, 0, 1, -ev.MoveStep*ev.CurActMag)
	ev.UpdateWorld()
}

// StepBackward moves Emer backward in current facing direction one step, and updates
func (ev *Env) StepBackward() {
	ev.Emer.Rel.MoveOnAxis(0, 0, 1, ev.MoveStep*ev.CurActMag)
	ev.UpdateWorld()
}

// RotBodyLeft rotates emer left and updates
func (ev *Env) RotBodyLeft() {
	ev.Emer.Rel.RotateOnAxis(0, 1, 0, ev.RotStep*ev.CurActMag)
	ev.UpdateWorld()
}

// RotBodyRight rotates emer right and updates
func (ev *Env) RotBodyRight() {
	ev.Emer.Rel.RotateOnAxis(0, 1, 0, -ev.RotStep*ev.CurActMag)
	ev.UpdateWorld()
}

// RotHeadLeft rotates head left and updates
func (ev *Env) RotHeadLeft() {
	ev.Head.Rel.RotateOnAxis(0, 1, 0, ev.RotStep*ev.CurActMag)
	ev.UpdateWorld()
}

// RotHeadRight rotates head right and updates
func (ev *Env) RotHeadRight() {
	ev.Head.Rel.RotateOnAxis(0, 1, 0, -ev.RotStep*ev.CurActMag)
	ev.UpdateWorld()
}

// RotHeadToBody rotates head straight
func (ev *Env) RotHeadToBody() {
	ev.Head.Rel.SetAxisRotation(0, 1, 0, 0)
	ev.UpdateWorld()
}

// RotBodyToHead rotates body to match current head rotation relative to body
func (ev *Env) RotBodyToHead() {
	aa := ev.Head.Rel.Quat.ToAxisAngle()
	ev.Head.Rel.SetAxisRotation(0, 1, 0, 0)
	ev.Emer.Rel.RotateOnAxis(0, 1, 0, mat32.RadToDeg(aa.W)) // just get angle assuming rest is up
	ev.UpdateWorld()
}

// TakeAction implements given action
func (ev *Env) TakeAction(act Actions) {
	switch act {
	case StepForward:
		ev.StepForward()
	// case StepBackward:
	// 	ev.StepBackward()
	case RotBodyLeft:
		ev.RotBodyLeft()
	case RotBodyRight:
		ev.RotBodyRight()
	case RotHeadLeft:
		ev.RotHeadLeft()
	case RotHeadRight:
		ev.RotHeadRight()
		// case RotHeadToBody:
		// 	ev.RotHeadToBody()
		// case RotBodyToHead:
		// 	ev.RotBodyToHead()
	}
}

// UpdateWorld updates world after action
func (ev *Env) UpdateWorld() {
	ev.World.UpdateWorld()
	if ev.View != nil {
		ev.View.UpdatePose()
		ev.UpdateState()
		ev.UpdateView()
	}
}

// NormHorizAng returns the normalized angle from quat, assuming a horizontal plane angle
// (i.e., axis = Y +-1)
func NormHorizAng(q mat32.Quat) float32 {
	aa := q.ToAxisAngle()
	ang := aa.W
	if aa.Y < 0 {
		ang = 2*mat32.Pi - ang
	}
	return ang / (2 * mat32.Pi)
}

// UpdateState updates the current state representations (depth, action)
func (ev *Env) UpdateState() {
	ev.View.Scene.ActivateWin()
	err := ev.View.RenderOffNode(&ev.Frame, ev.EyeR, &ev.Camera)
	if err != nil {
		log.Println(err)
		return
	}
	var depth []float32
	oswin.TheApp.RunOnMain(func() {
		tex := ev.Frame.Texture()
		tex.SetBotZero(true)
		ev.CurImage = tex.GrabImage()
		depth = ev.Frame.DepthAll()
	})
	copy(ev.RawDepth.Values, depth)
	evev.DepthNorm(&ev.CurDepth.Values, depth, &ev.Camera, false) // no flip!

	for ai := 0; ai < int(ActionsN); ai++ {
		off := ai * ev.ActRes
		sl := ev.CurActMap.Values[off : off+ev.ActRes]
		if ai == int(ev.CurAct) {
			ev.ActPop.Encode(&sl, ev.CurActMag, ev.ActRes)
		} else {
			nmag := ev.ActNoise + erand.Gauss(ev.ActNoise, -1)
			ev.ActPop.Encode(&sl, float32(nmag), ev.ActRes)
		}
	}

	ep := mat32.Vec2{ev.Emer.Abs.Pos.X, -ev.Emer.Abs.Pos.Z}
	ev.CurPos = ep.Div(mat32.Vec2{ev.Room.Width, ev.Room.Depth}).Add(mat32.Vec2{0.5, 0.5})
	// fmt.Printf("cur pos: %v\n", ev.CurPos)
	ev.PosPop.Encode(&ev.CurPosMap, ev.CurPos)

	maxStep := ev.RotStep / 360

	ev.HeadDir.Set(NormHorizAng(ev.Head.Abs.Quat))
	ev.AngPop.Encode(&ev.CurHeadDirMap.Values, ev.HeadDir.Cur, ev.AngRes)
	vel := mat32.Clamp(ev.HeadDir.Diff()/maxStep, -1, 1)
	ev.VelPop.Encode(&ev.CurHeadVelMap.Values, vel, ev.AngRes)

	ev.NeckAng.Set(NormHorizAng(ev.Head.Rel.Quat))
	ev.AngPop.Encode(&ev.CurNeckAngMap.Values, ev.NeckAng.Cur, ev.AngRes)
	vel = mat32.Clamp(ev.NeckAng.Diff()/maxStep, -1, 1)
	ev.VelPop.Encode(&ev.CurNeckAngVelMap.Values, vel, ev.AngRes)

	copy(ev.CurSomaMap.Values[0:ev.AngRes], ev.CurHeadDirMap.Values)
	copy(ev.CurSomaMap.Values[ev.AngRes:2*ev.AngRes], ev.CurHeadVelMap.Values)
	copy(ev.CurSomaMap.Values[2*ev.AngRes:3*ev.AngRes], ev.CurNeckAngMap.Values)
	copy(ev.CurSomaMap.Values[3*ev.AngRes:4*ev.AngRes], ev.CurNeckAngVelMap.Values)
}

// UpdateView updates view if gui active
func (ev *Env) UpdateView() {
	if ev.SnapImg != nil {
		ev.SnapImg.SetImage(ev.CurImage, 0, 0)
		ev.View.Scene.Render2D()
		ev.View.Scene.DirectWinUpload()
		ev.ViewDepth(ev.CurDepth.Values)
	}
}

// ViewDepth updates depth bitmap with depth data
func (ev *Env) ViewDepth(depth []float32) {
	cmap := giv.AvailColorMaps[string(ev.DepthMap)]
	ev.DepthImg.Resize(ev.Camera.Size)
	evev.DepthImage(ev.DepthImg.Pixels, depth, cmap, &ev.Camera)
	ev.DepthImg.UpdateSig()
}

// todo: need an offscreen render method that creates view for that case

func (ev *Env) OpenWindow() *gi.Window {
	width := 1024
	height := 768

	win := gi.NewWindow2D("navenv", "Navigation Environment", width, height, true) // true = pixel sizes
	ev.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()
	mfr.SetProp("spacing", units.NewEx(1))

	tbar := gi.AddNewToolBar(mfr, "main-tbar")
	tbar.SetStretchMaxWidth()
	tbar.Viewport = vp

	//////////////////////////////////////////
	//    Splitter

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X

	tvfr := gi.AddNewFrame(split, "tvfr", gi.LayoutHoriz)
	svfr := gi.AddNewFrame(split, "svfr", gi.LayoutHoriz)
	imfr := gi.AddNewFrame(split, "imfr", gi.LayoutHoriz)
	scfr := gi.AddNewFrame(split, "scfr", gi.LayoutHoriz)
	split.SetSplits(.1, .2, .2, .5)

	tv := giv.AddNewTreeView(tvfr, "tv")
	tv.SetRootNode(ev.World)

	sv := giv.AddNewStructView(svfr, "sv")
	sv.SetStretchMaxWidth()
	sv.SetStretchMaxHeight()
	sv.SetStruct(ev)

	tv.TreeViewSig.Connect(sv.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if data == nil {
			return
		}
		// tvr, _ := send.Embed(giv.KiT_TreeView).(*gi.TreeView) // root is sender
		tvn, _ := data.(ki.Ki).Embed(giv.KiT_TreeView).(*giv.TreeView)
		svr, _ := recv.Embed(giv.KiT_StructView).(*giv.StructView)
		if sig == int64(giv.TreeViewSelected) {
			svr.SetStruct(tvn.SrcNode)
		}
	})

	//////////////////////////////////////////
	//    Scene

	scvw := gi3d.AddNewSceneView(scfr, "sceneview")
	scvw.SetStretchMaxWidth()
	scvw.SetStretchMaxHeight()
	scvw.Config()
	sc := scvw.Scene()

	// first, add lights, set camera
	sc.BgColor.SetUInt8(230, 230, 255, 255) // sky blue-ish
	gi3d.AddNewAmbientLight(sc, "ambient", 0.3, gi3d.DirectSun)

	dir := gi3d.AddNewDirLight(sc, "dir", 1, gi3d.DirectSun)
	dir.Pos.Set(0, 2, 1) // default: 0,1,1 = above and behind us (we are at 0,0,X)

	ev.MakeView(sc)

	cam := &sc.Camera

	cam.Pose.Pos = mat32.Vec3{0, 20, 3.5}
	cam.LookAt(mat32.Vec3{0, 2, 0}, mat32.Vec3Y)
	sc.SaveCamera("3")

	cam.Pose.Pos.Set(0, 13.256021, 8.77191)
	cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-0.9999999, 0, -0}, 0.98724663)
	cam.Target.Set(0, 2.1378326, 1.4310836)
	cam.UpDir.Set(0, 0.86602557, -0.50000006)

	// cam.Pose.Pos = mat32.Vec3{0, 20, 30}
	// cam.LookAt(mat32.Vec3{0, 5, 0}, mat32.Vec3Y)
	sc.SaveCamera("2")

	cam.Pose.Pos = mat32.Vec3{-.86, .97, 2.7}
	cam.LookAt(mat32.Vec3{0, .8, 0}, mat32.Vec3Y)
	sc.SaveCamera("1")
	sc.SaveCamera("default")

	//////////////////////////////////////////
	//    Bitmap

	imfr.Lay = gi.LayoutVert
	gi.AddNewLabel(imfr, "lab-img", "Right Eye Image:")
	ev.SnapImg = gi.AddNewBitmap(imfr, "snap-img")
	ev.SnapImg.Resize(ev.Camera.Size)
	ev.SnapImg.LayoutToImgSize()
	ev.SnapImg.SetProp("vertical-align", gi.AlignTop)

	gi.AddNewLabel(imfr, "lab-depth", "Right Eye Depth:")
	ev.DepthImg = gi.AddNewBitmap(imfr, "depth-img")
	ev.DepthImg.Resize(ev.Camera.Size)
	ev.DepthImg.LayoutToImgSize()
	ev.DepthImg.SetProp("vertical-align", gi.AlignTop)

	tbar.AddAction(gi.ActOpts{Label: "Edit Env", Icon: "edit", Tooltip: "Edit the settings for the environment."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		sv.SetStruct(ev)
	})
	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize virtual world -- go back to starting positions etc."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.InitWorld()
	})
	tbar.AddAction(gi.ActOpts{Label: "Make", Icon: "update", Tooltip: "Re-make virtual world -- do this if you have changed any of the world parameters."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.ReMakeWorld()
	})
	tbar.AddAction(gi.ActOpts{Label: "Snap", Icon: "file-image", Tooltip: "Take a snapshot from perspective of the right eye of emer virtual robot."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.UpdateView()
	})
	tbar.AddSeparator("mv-sep")
	tbar.AddAction(gi.ActOpts{Label: "Fwd", Icon: "wedge-up", Tooltip: "Take a step Forward."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.StepForward()
	})
	tbar.AddAction(gi.ActOpts{Label: "Bkw", Icon: "wedge-down", Tooltip: "Take a step Backward."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.StepBackward()
	})
	tbar.AddAction(gi.ActOpts{Label: "Body Left", Icon: "wedge-left", Tooltip: "Rotate body left."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.RotBodyLeft()
	})
	tbar.AddAction(gi.ActOpts{Label: "Body Right", Icon: "wedge-right", Tooltip: "Rotate body right."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.RotBodyRight()
	})
	tbar.AddAction(gi.ActOpts{Label: "Head Left", Icon: "wedge-left", Tooltip: "Rotate body left."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.RotHeadLeft()
	})
	tbar.AddAction(gi.ActOpts{Label: "Head Right", Icon: "wedge-right", Tooltip: "Rotate body right."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.RotHeadRight()
	})
	tbar.AddAction(gi.ActOpts{Label: "Body To Head", Icon: "update", Tooltip: "Rotate body to match head orientation relative to body."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.RotBodyToHead()
	})
	tbar.AddAction(gi.ActOpts{Label: "Head To Body", Icon: "update", Tooltip: "Rotate head back to match body."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.RotHeadToBody()
	})
	// tbar.AddSeparator("rm-sep")
	// tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Open browser on README."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 	gi.OpenURL("https://github.com/emer/eve/blob/master/examples/virtroom/README.md")
	// })

	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	win.MainMenuUpdated()
	vp.UpdateEndNoSig(updt)
	return win
}
