package main

// TODO Move to emergent/egui

import (
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/relpos"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/mat32"
	"math"
	"math/rand"
)

// UserInterface automatically handles creation of the GUI if requested, otherwise runs on the command line.
type UserInterface struct {
	Looper        *looper.Manager
	Network       emer.Network
	Logs          *elog.Logs
	Stats         *estats.Stats
	GUI           *egui.GUI `desc:"More directly handles graphical elements."`
	AppName       string
	AppAbout      string
	AppTitle      string
	StructForView interface{} `desc:"This might be Sim or any other object you want to display to the user."`

	// Callbacks
	InitCallback func()

	// Internal
	guiEnabled bool
}

func AddDefaultGUICallbacks(manager *looper.Manager, gui *egui.GUI) {
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
	ui.CreateAndRunGuiWithAdditionalConfig(func() {})
}

func (ui *UserInterface) CreateAndRunGuiWithAdditionalConfig(config func()) {
	if ui.GUI == nil {
		ui.GUI = &egui.GUI{}
	}
	ui.guiEnabled = true

	AddDefaultGUICallbacks(ui.Looper, ui.GUI)

	gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
		title := ui.AppTitle
		//cat := struct{ Name string }{Name: "Cat"}
		ui.GUI.MakeWindow(ui.StructForView, ui.AppName, title, ui.AppAbout)
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
		} // ui.Looper.Stacks.Keys()
		ui.GUI.AddLooperCtrl(ui.Looper, modes)

		// Run custom code to configure the GUI.
		config()

		ui.GUI.FinalizeGUI(false)

		ui.GUI.Win.StartEventLoop()
	})
}

func (ui *UserInterface) RunWithoutGui() {
	// TODO Something something command line here?
	ui.Looper.Run()
}

// TODO All of the following is for automatically placing layers. It works OK but it's probably overcomplicated.
func computeLayerOverlap(lay1 emer.Layer, lay2 emer.Layer) float32 {
	s1 := lay1.Size()
	s2 := lay2.Size()
	// Overlap in X
	xo := float32(0)
	xlow := math.Max(float64(lay1.Pos().X), float64(lay2.Pos().X))
	xhigh := math.Max(float64(lay1.Pos().X+s1.X), float64(lay2.Pos().X+s2.X))
	if xhigh > xlow {
		xo = float32(xhigh - xlow)
	}
	// Overlap in Z
	zo := float32(0)
	zlow := math.Max(float64(lay1.Pos().Z), float64(lay2.Pos().Z))
	zhigh := math.Max(float64(lay1.Pos().Z+s1.Y), float64(lay2.Pos().Z+s2.Y)) * 2
	if zhigh > zlow {
		zo = float32(zhigh - zlow)
	}
	// Overlap is the product of overlap in both dimensions.
	return xo * zo
}

func scoreNet(net emer.Network, pctDone float32) float32 {
	score := float32(0)
	idealDist := float32(5)
	idealDistUnconnect := float32(15)
	unconnectedTerm := float32(0.1)
	inOutPosTerm := float32(100)
	negativeTerm := float32(10000)                 // This should be much bigger than inOutPosTerm
	overlapTerm := float32(10) * pctDone * pctDone // Care a lot more about overlap as we go
	for i := 0; i < net.NLayers(); i++ {
		layer := net.Layer(i)
		// Connected layers about the right distance apart.
		connectedLayers := map[emer.Layer]bool{}
		for j := 0; j < net.Layer(i).NSendPrjns(); j++ {
			recLayer := layer.SendPrjn(j).RecvLay()
			connectedLayers[recLayer] = true
			pos1 := layer.Pos()
			pos2 := recLayer.Pos()
			dist := mat32.Sqrt((pos1.X-pos2.X)*(pos1.X-pos2.X) + (pos1.Y-pos2.Y)*(pos1.Y-pos2.Y))
			score += (dist - idealDist) * (dist - idealDist)
		}
		// Other layers a good distance away too
		for j := 0; j < net.NLayers(); j++ {
			_, ok := connectedLayers[net.Layer(j)]
			if !ok {
				pos1 := layer.Pos()
				pos2 := net.Layer(j).Pos()
				dist := mat32.Sqrt((pos1.X-pos2.X)*(pos1.X-pos2.X) + (pos1.Y-pos2.Y)*(pos1.Y-pos2.Y))
				score += (dist - idealDistUnconnect) * (dist - idealDistUnconnect) * unconnectedTerm
			}
		}
		// No overlap.
		for j := 0; j < net.NLayers(); j++ {
			if i != j {
				score -= computeLayerOverlap(layer, net.Layer(j)) * overlapTerm
			}
		}
		// Inputs to the bottom, outputs to the top.
		if layer.Type() == emer.Input {
			score -= layer.Pos().Z * inOutPosTerm
		}
		if layer.Type() == emer.Target {
			score += layer.Pos().Z * inOutPosTerm
		}
		// Don't go negative.
		if layer.Pos().Z < 0 {
			score += layer.Pos().Z * negativeTerm
		}
		//if layer.Pos().X < 0 {
		//	score += layer.Pos().X * negativeTerm
		//}
	}
	return score
}

// PositionNetworkLayersAutomatically tries to find a configuration for the network layers where they're close together, but not overlapping. It tries to put connected layers closer together, input layers near the bottom, and target layers near the top. It uses a random walk algorithm that randomly permutes the network and only keeps permutations if they improve the network's overall configuration score.
// numSettlingIterations is the number of random moves it tries for each layer. Larger values will generally get better results but compute time grows linearly.
func PositionNetworkLayersAutomatically(net emer.Network, numSettlingIterations int) {
	size := float32(50) // The size of the positioning area
	wiggleSize := float32(5)
	// Initially randomize layers
	for j := 0; j < net.NLayers(); j++ {
		layer := net.Layer(int(j))
		layer.SetRelPos(relpos.Rel{Rel: relpos.NoRel})
		layer.SetPos(mat32.Vec3{rand.Float32() * size, 0, rand.Float32() * size})
	}
	for i := 0; i < numSettlingIterations; i++ {
		for j := 0; j < net.NLayers(); j++ {
			layer := net.Layer(int(j))
			pos := layer.Pos()
			// Make a random change and see if it improves things.
			offset := mat32.Vec3{rand.Float32()*wiggleSize - wiggleSize/2, 0, rand.Float32()*wiggleSize - wiggleSize/2}
			beforeScore := scoreNet(net, float32(i)/float32(numSettlingIterations))
			newPos := mat32.Vec3{pos.X + offset.X, pos.Y + offset.Y, pos.Z + offset.Z}
			layer.SetPos(newPos)
			afterScore := scoreNet(net, float32(i)/float32(numSettlingIterations))
			if beforeScore > afterScore {
				// Revert this random change.
				layer.SetPos(pos)
			}
		}
		// Simulated annealing.
		wiggleSize = wiggleSize * (1 - 1/float32(numSettlingIterations))
	}
}
