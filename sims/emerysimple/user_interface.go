package main

// TODO Move to emergent/

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/mat32"
)

// UserInterface automatically handles creation of the GUI if requested, otherwise runs on the command line.
type UserInterface struct {
	Looper        *looper.LoopManager
	Network       *axon.Network // TODO Is there a general interface for all these?
	Logs          *elog.Logs
	Stats         *estats.Stats
	GUI           *egui.GUI `desc:"More directly handles graphical elements."`
	AppName       string
	AppAbout      string
	AppTitle      string
	StructForView *Sim `desc:"This might be Sim."` // TODO Make interface{}. Ask Randy how this interacts with reflection.

	// Callbacks
	InitCallback func()

	// Internal
	guiEnabled bool
}

func (ui UserInterface) Init() *UserInterface {
	return &ui
}

func AddDefaultGUICallbacks(manager *looper.LoopManager, gui *egui.GUI) {
	for _, m := range []etime.Modes{etime.Train} {
		curMode := m // For closures.
		for _, t := range []etime.Times{etime.Trial} {
			curTime := t
			if manager.GetLoop(curMode, curTime).OnEnd.HasNameLike("UpdateNetView") {
				// There might be a case where another function also Updates the NetView, and we don't want to do it twice. In particular, Net.WtFmDWt clears some values at the end of Trial, and it wants to update the view before doing so.
				continue
			}
			manager.GetLoop(curMode, curTime).OnEnd.Add("GUI:UpdateNetView", func() {
				gui.UpdateNetView() // TODO Use update timescale variable
			})
		}
	}
}

// EnableGui creates a GUI, with which the user can control the application. It will loop forever.
func (ui *UserInterface) CreateAndRunGui() {
	ui.guiEnabled = true

	AddDefaultGUICallbacks(ui.Looper, ui.GUI)

	gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
		title := ui.AppTitle
		//cat := struct{ Name string }{Name: "Cat"}
		ui.GUI.MakeWindow(&ui.StructForView, ui.AppName, title, ui.AppAbout)
		ui.GUI.CycleUpdateInterval = 10

		if ui.Network != nil {
			nv := ui.GUI.AddNetView("NetView")
			nv.Params.MaxRecs = 300
			nv.SetNet(ui.Network)
			ui.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
			ui.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
		}

		if ui.Logs != nil {
			ui.GUI.AddPlots(title, ui.Logs)
		}

		if ui.Stats != nil && len(ui.Stats.Rasters) > 0 {
			stb := ui.GUI.TabView.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
			stb.Lay = gi.LayoutVert
			stb.SetStretchMax()
			for _, lnm := range ui.Stats.Rasters {
				sr := ui.Stats.F32Tensor("Raster_" + lnm)
				ui.GUI.ConfigRasterGrid(stb, lnm, sr)
			}
		}

		ui.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ui.InitCallback()
				ui.GUI.UpdateWindow()
			},
		})

		modes := []etime.Modes{}
		for m, _ := range ui.Looper.Stacks {
			modes = append(modes, m)
		} // Agh why you no have .keys
		ui.GUI.AddLooperCtrl(ui.Looper, modes)

		ui.GUI.FinalizeGUI(false)

		ui.GUI.Win.StartEventLoop()
	})
}

// TODO What does this do if GUI is enabled?
func (ui *UserInterface) RunWithoutGui() {
	// TODO Something something command line here?
	ui.Looper.Steps.Run()
}