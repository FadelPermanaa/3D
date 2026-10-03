"""Procedural McLaren Senna-style hypercar for Blender (bpy 4.2+ / 5.x).

Builds the car from lofted cross-sections, cuts arches/intakes/vents with
booleans, then renders previews and exports GLB, FBX (Roblox-ready) and .blend.

    python build_supercar.py [out_dir] [--preview] [--no-render]

No logos or badges are modelled: it is an original design in that style.
"""
import math
import os
import sys

import bpy  # must precede bmesh
import bmesh
import numpy as np
from mathutils import Matrix, Vector

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = os.path.abspath(ARGS[0] if ARGS else os.path.join(os.path.dirname(__file__), "out"))
PREVIEW = "--preview" in sys.argv
RENDER = "--no-render" not in sys.argv

# ---- Tweakables -------------------------------------------------------------
PAINT = (0.07, 0.17, 0.42)          # Senna-style metallic blue
LENGTH = 4.55                       # metres, nose to tail
WHEELBASE = 2.67
FRONT_AXLE_Y = -1.36                # front of car points to -Y
R_FRONT, R_REAR = 0.335, 0.355      # tyre radius
W_FRONT, W_REAR = 0.25, 0.30        # tyre width
TRACK_FRONT, TRACK_REAR = 0.83, 0.80
ROBLOX_TRI_LIMIT = 19500            # per MeshPart, with margin under 20k

os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


# ---- Materials --------------------------------------------------------------
def material(name, color, metallic=0.0, rough=0.5, coat=0.0, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    b.inputs["Coat Weight"].default_value = coat
    if emit:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emit
    m.diffuse_color = (*color, 1)
    return m


M = {
    "paint": material("Paint", PAINT, metallic=0.7, rough=0.28, coat=1.0),
    "black": material("GlossBlack", (0.012, 0.012, 0.013), rough=0.18, coat=0.6),
    "trim": material("SatinBlack", (0.02, 0.02, 0.022), rough=0.55),
    "carbon": material("Carbon", (0.03, 0.032, 0.035), metallic=0.3, rough=0.35, coat=0.8),
    "glass": material("Glass", (0.015, 0.02, 0.025), metallic=0.1, rough=0.03, coat=1.0),
    "tyre": material("Tyre", (0.025, 0.025, 0.025), rough=0.85),
    "rim": material("Rim", (0.62, 0.63, 0.65), metallic=1.0, rough=0.22),
    "disc": material("BrakeDisc", (0.35, 0.34, 0.33), metallic=1.0, rough=0.45),
    "caliper": material("Caliper", (1.0, 0.32, 0.02), rough=0.3, coat=0.5),
    "chrome": material("Chrome", (0.8, 0.8, 0.82), metallic=1.0, rough=0.08),
    "headlight": material("Headlight", (1.0, 0.97, 0.9), emit=6.0),
    "taillight": material("Taillight", (1.0, 0.02, 0.01), emit=8.0),
    "lens": material("LightHousing", (0.05, 0.05, 0.055), metallic=0.8, rough=0.15),
}


# ---- Helpers ----------------------------------------------------------------
def to_object(name, bm, mat, smooth=True, sharp_angle=40.0):
    me = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    if mat:
        me.materials.append(mat)
    if smooth:
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(sharp_angle))
    return ob


def bake(ob):
    """Apply the whole modifier stack into the mesh."""
    dg = bpy.context.evaluated_depsgraph_get()
    new = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = new
    bpy.data.meshes.remove(old)


def box(bm, center, size, rot=(0, 0, 0)):
    R = Matrix.Rotation(rot[2], 4, "Z") @ Matrix.Rotation(rot[1], 4, "Y") @ Matrix.Rotation(rot[0], 4, "X")
    S = Matrix.Diagonal((*size, 1))
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation(center) @ R @ S)


def cylinder_x(bm, center, radius, depth, segments=48):
    M_ = Matrix.Translation(center) @ Matrix.Rotation(math.pi / 2, 4, "Y")
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=depth, matrix=M_)


def ellipsoid(bm, center, size, rot=(0, 0, 0), segs=24):
    R = Matrix.Rotation(rot[2], 4, "Z") @ Matrix.Rotation(rot[1], 4, "Y") @ Matrix.Rotation(rot[0], 4, "X")
    S = Matrix.Diagonal((*size, 1))
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=segs // 2, radius=1.0, matrix=Matrix.Translation(center) @ R @ S)


def single(name, mat, builder, smooth=True):
    bm = bmesh.new()
    builder(bm)
    return to_object(name, bm, mat, smooth=smooth)


def curve(keys, t):
    """Piecewise-linear keys, Gaussian-smoothed on a dense grid, sampled at t."""
    k = np.array(keys, dtype=float)
    dense = np.linspace(0, 1, 801)
    v = np.interp(dense, k[:, 0], k[:, 1])
    ker = np.exp(-np.linspace(-3, 3, 41) ** 2)
    v = np.convolve(np.pad(v, 20, mode="edge"), ker / ker.sum(), "valid")
    return np.interp(t, dense, v)


def loft(name, mat, t, Y, W, B, H, n_top, n_bot, tumble, ring=48, dip=None):
    """Closed tube of superellipse sections; ends are poked so subdivision stays clean."""
    bm = bmesh.new()
    rings = []
    for i in range(len(t)):
        c, hh = (B[i] + H[i]) / 2, max((H[i] - B[i]) / 2, 1e-4)
        rv = []
        for j in range(ring):
            th = 2 * math.pi * j / ring
            cs, sn = math.cos(th), math.sin(th)
            n = n_top if sn >= 0 else n_bot
            x = W[i] * math.copysign(abs(cs) ** (2 / n), cs)
            z = hh * math.copysign(abs(sn) ** (2 / n), sn)
            if z > 0:
                x *= 1 - tumble * (z / hh) ** 2
            zz = c + z
            if dip is not None and z > 0:
                zz -= dip[i] * math.exp(-((x / (0.5 * W[i])) ** 2)) * (z / hh) ** 6
            rv.append(bm.verts.new((x, Y[i], zz)))
        rings.append(rv)
    for i in range(len(rings) - 1):
        for j in range(ring):
            bm.faces.new((rings[i][j], rings[i][(j + 1) % ring], rings[i + 1][(j + 1) % ring], rings[i + 1][j]))
    caps = [bm.faces.new(rings[0]), bm.faces.new(rings[-1])]
    bmesh.ops.poke(bm, faces=caps)
    return to_object(name, bm, mat)


def cutter_collection(name, builders):
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    for cname, mat, fn in builders:
        ob = single(cname, mat, fn, smooth=False)
        scene.collection.objects.unlink(ob)
        col.objects.link(ob)
    return col


def boolean_cut(ob, col):
    mod = ob.modifiers.new("cuts", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.operand_type = "COLLECTION"
    mod.collection = col
    mod.solver = "EXACT"
    mod.material_mode = "TRANSFER"
    return mod


def remove_collection(col):
    for ob in list(col.objects):
        bpy.data.meshes.remove(ob.data)
    bpy.data.collections.remove(col)


# ---- Body -------------------------------------------------------------------
u = np.linspace(0, 1, 72)
t = 0.5 - 0.5 * np.cos(np.pi * u)  # denser at nose and tail
Y = -LENGTH / 2 + t * LENGTH

body_W = curve([(0, 0.52), (0.02, 0.80), (0.07, 0.90), (0.14, 0.955), (0.20, 0.965), (0.27, 0.93),
                (0.38, 0.875), (0.48, 0.865), (0.60, 0.92), (0.72, 0.97), (0.84, 0.97), (0.95, 0.94),
                (0.99, 0.90), (1.0, 0.82)], t)
body_B = curve([(0, 0.26), (0.02, 0.16), (0.07, 0.115), (0.85, 0.11), (0.93, 0.16), (0.985, 0.28), (1.0, 0.36)], t)
body_H = curve([(0, 0.36), (0.02, 0.45), (0.07, 0.55), (0.14, 0.655), (0.21, 0.74), (0.28, 0.745),
                (0.38, 0.80), (0.50, 0.86), (0.62, 0.90), (0.74, 0.95), (0.84, 0.955), (0.94, 0.93),
                (0.99, 0.90), (1.0, 0.82)], t)
hood_dip = curve([(0, 0), (0.05, 0.0), (0.1, 0.035), (0.24, 0.035), (0.30, 0.0), (0.72, 0.0), (0.82, 0.03),
                  (0.95, 0.03), (1, 0)], t)

body = loft("Body", M["paint"], t, Y, body_W, body_B, body_H, n_top=3.2, n_bot=7.0, tumble=0.10, dip=hood_dip)
sub = body.modifiers.new("subd", "SUBSURF")
sub.levels = sub.render_levels = 2


def y_at(tt):
    return -LENGTH / 2 + tt * LENGTH


def body_top(tt):
    return float(np.interp(tt, t, body_H))


REAR_AXLE_Y = FRONT_AXLE_Y + WHEELBASE


def arch(y, r):
    def fn(bm):
        for s in (-1, 1):
            cylinder_x(bm, (s * 1.0, y, r), r + 0.04, 0.95, segments=64)
    return fn


def side_scoops(bm):
    for s in (-1, 1):
        ellipsoid(bm, (s * 1.02, 0.58, 0.56), (0.22, 0.55, 0.19), rot=(math.radians(-8), 0, 0))


def door_glass(bm):
    for s in (-1, 1):
        ellipsoid(bm, (s * 0.94, -0.40, 0.45), (0.09, 0.40, 0.10), rot=(math.radians(4), 0, 0))


def front_intakes(bm):
    for s in (-1, 1):
        ellipsoid(bm, (s * 0.62, -2.30, 0.25), (0.28, 0.26, 0.095), rot=(0, s * math.radians(-6), s * math.radians(-12)))
    box(bm, (0, -2.33, 0.20), (0.56, 0.30, 0.075))


def headlight_sockets(bm):
    for s in (-1, 1):
        ellipsoid(bm, (s * 0.62, -2.02, 0.50), (0.32, 0.26, 0.085), rot=(math.radians(-14), s * math.radians(10), s * math.radians(28)), segs=8)


def hood_vents(bm):
    for s in (-1, 1):
        # large louvred openings on top of the front fenders
        box(bm, (s * 0.58, -1.80, 0.70), (0.36, 0.42, 0.12), rot=(math.radians(8), s * math.radians(14), s * math.radians(-18)))
    box(bm, (0, -1.95, 0.64), (0.40, 0.30, 0.10), rot=(math.radians(12), 0, 0))    # nose duct


def rear_openings(bm):
    box(bm, (0, 2.36, 0.64), (1.55, 0.30, 0.26))           # rear mesh grille
    box(bm, (0, 2.36, 0.24), (1.40, 0.30, 0.16))           # diffuser mouth
    for s in (-1, 1):                                      # rear arch vents
        box(bm, (s * 0.96, 1.85, 0.62), (0.20, 0.40, 0.16), rot=(0, 0, s * math.radians(25)))


cuts = cutter_collection("BodyCuts", [
    ("cut_arch_f", M["trim"], arch(FRONT_AXLE_Y, R_FRONT)),
    ("cut_arch_r", M["trim"], arch(REAR_AXLE_Y, R_REAR)),
    ("cut_scoops", M["trim"], side_scoops),
    ("cut_door_glass", M["glass"], door_glass),
    ("cut_intakes", M["trim"], front_intakes),
    ("cut_lights", M["lens"], headlight_sockets),
    ("cut_vents", M["trim"], hood_vents),
    ("cut_rear", M["trim"], rear_openings),
])
boolean_cut(body, cuts)
bake(body)
remove_collection(cuts)
body.data.set_sharp_from_angle(angle=math.radians(40))


CARBON_Z, NOSE_Y, TAIL_Y = 0.31, -1.75, 1.75


def carbon_line(y):
    """Height below which the body is bare carbon; rises toward the nose and the tail."""
    return CARBON_Z + 0.36 * max(0.0, -y + NOSE_Y) + 0.25 * max(0.0, y - TAIL_Y)


def zone_materials(ob):
    """Senna-style two-tone: carbon lower body and a dark nose centre, split on clean cut lines."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    planes = [
        ((0, 0, CARBON_Z), (0, 0, 1)),
        ((0, NOSE_Y, CARBON_Z), (0, 0.36, 1)),
        ((0, TAIL_Y, CARBON_Z), (0, -0.25, 1)),
        ((0.30, 0, 0), (1, 0, 0)),
        ((-0.30, 0, 0), (1, 0, 0)),
        ((0, -1.55, 0), (0, 1, 0)),
    ]
    for co, no in planes:
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=no)
    names = [m.name for m in me.materials]
    if M["carbon"].name not in names:
        me.materials.append(M["carbon"])
        names.append(M["carbon"].name)
    paint, carbon = names.index("Paint"), names.index(M["carbon"].name)
    for f in bm.faces:
        if f.material_index != paint:
            continue
        c = f.calc_center_median()
        if c.z < carbon_line(c.y) or (f.normal.z > 0.35 and abs(c.x) < 0.30 and c.y < -1.55):
            f.material_index = carbon
    bm.to_mesh(me)
    bm.free()


zone_materials(body)

# ---- Cabin / greenhouse -----------------------------------------------------
gt = np.linspace(0.27, 0.79, 48)
gY = -LENGTH / 2 + gt * LENGTH
k = lambda tt: body_top(tt)
g_top = curve([(0.27, k(0.27) - 0.02), (0.30, k(0.30) + 0.04), (0.36, 0.93), (0.44, 1.10), (0.51, 1.165),
               (0.58, 1.16), (0.65, 1.10), (0.72, 1.02), (0.79, k(0.79) - 0.02)], gt)
g_bot = np.array([k(x) - 0.30 for x in gt])
g_top = np.maximum(g_top, g_bot + 0.05)
g_W = curve([(0.27, 0.40), (0.31, 0.62), (0.40, 0.73), (0.52, 0.74), (0.64, 0.70), (0.74, 0.58), (0.79, 0.35)],
            gt)
cabin = loft("Cabin", M["glass"], gt, gY, g_W, g_bot, g_top, n_top=2.6, n_bot=4.0, tumble=0.42, ring=40)
sub = cabin.modifiers.new("subd", "SUBSURF")
sub.levels = sub.render_levels = 2
bake(cabin)
cabin.data.set_sharp_from_angle(angle=math.radians(50))

def roof_scoop(bm):
    tt = 0.60
    z = float(np.interp(tt, gt, g_top))
    ellipsoid(bm, (0, y_at(tt), z + 0.005), (0.12, 0.40, 0.045), segs=24)


def roof_scoop_mouth(bm):
    tt = 0.555
    z = float(np.interp(tt, gt, g_top))
    ellipsoid(bm, (0, y_at(tt), z + 0.025), (0.08, 0.03, 0.025), segs=16)


single("RoofScoop", M["paint"], roof_scoop)
single("RoofScoopIntake", M["black"], roof_scoop_mouth)

# ---- Lights -----------------------------------------------------------------
def headlights(bm):
    for s in (-1, 1):
        # two LED blades set inside each socket, aligned with it
        rot = (math.radians(-14), s * math.radians(10), s * math.radians(28))
        box(bm, (s * 0.63, -2.03, 0.52), (0.24, 0.03, 0.012), rot=rot)      # upper LED blade
        box(bm, (s * 0.58, -2.05, 0.485), (0.15, 0.03, 0.010), rot=rot)      # lower LED blade


single("Headlights", M["headlight"], headlights)


def taillights(bm):
    for s in (-1, 1):
        box(bm, (s * 0.52, 2.255, 0.79), (0.62, 0.04, 0.025))
        box(bm, (s * 0.83, 2.215, 0.68), (0.03, 0.05, 0.22), rot=(0, s * math.radians(10), 0))
    box(bm, (0, 2.25, 0.835), (0.30, 0.03, 0.018))    # centre brake light


single("Taillights", M["taillight"], taillights, smooth=False)

# ---- Aero & trim --------------------------------------------------------------
def splitter(bm):
    box(bm, (0, -2.06, 0.08), (1.74, 0.58, 0.024))
    for s in (-1, 1):
        box(bm, (s * 0.80, -1.98, 0.14), (0.018, 0.34, 0.09))   # splitter end fences


def side_skirts(bm):
    for s in (-1, 1):
        box(bm, (s * 0.90, (FRONT_AXLE_Y + REAR_AXLE_Y) / 2, 0.13), (0.10, 1.85, 0.05))


def diffuser(bm):
    pitch = math.radians(-14)
    box(bm, (0, 1.92, 0.15), (1.40, 0.62, 0.022), rot=(pitch, 0, 0))
    for x in (-0.55, -0.27, 0.0, 0.27, 0.55):
        box(bm, (x, 1.95, 0.19), (0.018, 0.55, 0.15), rot=(pitch, 0, 0))


def rear_mesh(bm):
    box(bm, (0, 2.25, 0.64), (1.52, 0.02, 0.24))


def canards(bm):
    for s in (-1, 1):
        for z, yy in ((0.30, -2.07), (0.40, -2.00)):
            box(bm, (s * 0.86, yy + 0.12, z), (0.18, 0.22, 0.012), rot=(math.radians(-6), s * math.radians(-10), s * math.radians(-25)))


single("Splitter", M["carbon"], splitter, smooth=False)
single("Canards", M["carbon"], canards, smooth=False)
single("SideSkirts", M["carbon"], side_skirts, smooth=False)
single("Diffuser", M["carbon"], diffuser, smooth=False)
single("RearGrille", M["trim"], rear_mesh, smooth=False)


def exhausts(bm):
    for s in (-1, 1):
        cyl = Matrix.Translation((s * 0.10, 1.98, 0.97)) @ Matrix.Rotation(math.radians(-35), 4, "X")
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.055, radius2=0.06, depth=0.16, matrix=cyl)


single("Exhausts", M["chrome"], exhausts)


def airfoil_wing(bm, span=1.80, chord=0.42, thick=0.10, camber=0.06, n=20):
    """NACA 4-digit style section (chord along +Y), extruded along X."""
    upper, lower = [], []
    for i in range(n + 1):
        x = (1 - math.cos(math.pi * i / n)) / 2
        yt = 5 * thick * (0.2969 * math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1015 * x ** 4)
        yc = camber * (0.8 * x - x * x) / 0.16 if x < 0.4 else camber * (0.2 + 0.8 * x - x * x) / 0.36
        upper.append((x, yc + yt))
        lower.append((x, yc - yt))
    loop = upper + lower[-2:0:-1]
    sides = []
    for sx in (-span / 2, span / 2):
        sides.append([bm.verts.new((sx, x * chord, z * chord)) for x, z in loop])
    m = len(loop)
    for j in range(m):
        bm.faces.new((sides[0][j], sides[0][(j + 1) % m], sides[1][(j + 1) % m], sides[1][j]))
    bm.faces.new(sides[0])
    bm.faces.new(sides[1])


wing = single("RearWing", M["carbon"], airfoil_wing)
wing.data.set_sharp_from_angle(angle=math.radians(30))
wing.rotation_euler = (math.radians(12), 0, 0)
wing.location = (0, 1.80, 1.24)


def wing_parts(bm):
    for s in (-1, 1):
        box(bm, (s * 0.905, 2.0, 1.25), (0.014, 0.48, 0.17))                        # endplates
        box(bm, (s * 0.30, 2.02, 1.16), (0.03, 0.12, 0.30), rot=(math.radians(-22), 0, 0))  # swan-neck pylons
        box(bm, (s * 0.30, 2.07, 1.30), (0.03, 0.18, 0.04))                            # pylon hook over the wing


single("WingSupports", M["carbon"], wing_parts, smooth=False)


def mirrors(bm):
    for s in (-1, 1):
        # stalk rises from the door top to the pod
        box(bm, (s * 0.80, -0.66, 0.78), (0.035, 0.05, 0.16), rot=(0, s * math.radians(-40), 0))
        ellipsoid(bm, (s * 0.90, -0.67, 0.82), (0.10, 0.07, 0.05), segs=20)


single("Mirrors", M["carbon"], mirrors)


# ---- Wheels -----------------------------------------------------------------
def wheel(name, y, side, r, width, track):
    """One rotating assembly (tyre, rim, spokes, disc) with origin at the hub."""
    bm = bmesh.new()
    s = side
    rim_r = r * 0.72
    # tyre: superellipse profile revolved about X
    segs, prof_n = 72, 24
    cr, hr, ha = (r + rim_r) / 2, (r - rim_r) / 2 + 0.002, width / 2
    rings = []
    for i in range(segs):
        phi = 2 * math.pi * i / segs
        rv = []
        for j in range(prof_n):
            a = 2 * math.pi * j / prof_n
            rad = cr + hr * math.copysign(abs(math.cos(a)) ** 0.5, math.cos(a))
            ax = ha * math.copysign(abs(math.sin(a)) ** 0.5, math.sin(a))
            rv.append(bm.verts.new((ax, rad * math.cos(phi), rad * math.sin(phi))))
        rings.append(rv)
    tyre_faces = []
    for i in range(segs):
        for j in range(prof_n):
            tyre_faces.append(bm.faces.new((rings[i][j], rings[i][(j + 1) % prof_n],
                                            rings[(i + 1) % segs][(j + 1) % prof_n], rings[(i + 1) % segs][j])))
    for f in tyre_faces:
        f.material_index = 0
    n_before = len(bm.faces)
    # rim barrel + lip
    M_ = Matrix.Translation((-s * 0.01, 0, 0)) @ Matrix.Rotation(math.pi / 2, 4, "Y")
    bmesh.ops.create_cone(bm, cap_ends=False, segments=64, radius1=rim_r * 0.99, radius2=rim_r * 0.99,
                          depth=width * 0.92, matrix=M_)
    lip = Matrix.Translation((s * (width / 2 - 0.012), 0, 0)) @ Matrix.Rotation(math.pi / 2, 4, "Y")
    bmesh.ops.create_cone(bm, cap_ends=False, segments=64, radius1=rim_r * 1.02, radius2=rim_r * 1.02,
                          depth=0.02, matrix=lip)
    # 5 twin spokes, dished outward
    for k_ in range(10):
        for off in (0.0,):
            ang = 2 * math.pi * k_ / 10 + off
            Mt = (Matrix.Translation((s * (width / 2 - 0.03), 0, 0)) @ Matrix.Rotation(ang, 4, "X")
                  @ Matrix.Translation((0, 0, rim_r * 0.52)) @ Matrix.Rotation(-s * math.radians(8), 4, "Y")
                  @ Matrix.Diagonal((0.03, 0.026, rim_r * 0.94, 1)))
            bmesh.ops.create_cube(bm, size=1.0, matrix=Mt)
    cylinder_x(bm, (s * (width / 2 - 0.025), 0, 0), 0.075, 0.05, segments=32)
    bm.faces.ensure_lookup_table()
    for f in bm.faces[n_before:]:
        f.material_index = 1
    n_before = len(bm.faces)
    # brake disc (rotates with wheel)
    cylinder_x(bm, (s * 0.0, 0, 0), rim_r * 0.84, 0.032, segments=64)
    bm.faces.ensure_lookup_table()
    for f in bm.faces[n_before:]:
        f.material_index = 2
    me = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    for m_ in (M["tyre"], M["rim"], M["disc"]):
        me.materials.append(m_)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(35))
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    ob.location = (side * track, y, r)
    return ob


wheels = []
for label, y, r, w, tr in (("F", FRONT_AXLE_Y, R_FRONT, W_FRONT, TRACK_FRONT),
                           ("R", REAR_AXLE_Y, R_REAR, W_REAR, TRACK_REAR)):
    for side, sname in ((1, "R"), (-1, "L")):
        wheels.append(wheel(f"Wheel_{label}{sname}", y, side, r, w, tr))


def calipers(bm):
    for y, r, tr in ((FRONT_AXLE_Y, R_FRONT, TRACK_FRONT), (REAR_AXLE_Y, R_REAR, TRACK_REAR)):
        for s in (-1, 1):
            rim_r = r * 0.72
            for a in range(5):
                ang = math.radians(25 + a * 11)    # arc at the trailing top of the disc
                c = Vector((s * tr, y + math.sin(ang) * rim_r * 0.78, r + math.cos(ang) * rim_r * 0.78))
                box(bm, c, (0.07, 0.07, 0.06), rot=(-ang, 0, 0))


single("Calipers", M["caliper"], calipers, smooth=False)

CAR = [o for o in scene.objects if o.type == "MESH"]


# ---- Stats ------------------------------------------------------------------
def tris(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


print("Triangles per object:")
for o in CAR:
    print(f"  {o.name:14s} {tris(o):7d}")
print(f"  {'TOTAL':14s} {sum(tris(o) for o in CAR):7d}")

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "supercar.blend"))


def select_only(objs):
    for o in scene.objects:
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = objs[0]


select_only(CAR)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "supercar.glb"), export_format="GLB",
                          use_selection=True, export_apply=True)

# ---- Render -----------------------------------------------------------------
if RENDER:
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.6, 0.66, 1)
    bg.inputs["Strength"].default_value = 0.30
    scene.world = world

    bpy.ops.mesh.primitive_plane_add(size=60)
    ground = bpy.context.active_object
    ground.data.materials.append(material("Ground", (0.16, 0.165, 0.175), rough=0.4))

    def area(name, loc, energy, size):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size = energy, size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (-(Vector((0, 0, 0.5)) - Vector(loc))).to_track_quat("Z", "Y").to_euler()
        scene.collection.objects.link(lo)

    area("Key", (4.0, -5.0, 6.0), 1800, 6.0)
    area("Fill", (-6.0, -2.0, 3.5), 500, 8.0)
    area("Rim", (-2.0, 7.0, 5.0), 1500, 6.0)
    area("Top", (0, 0, 8.0), 1000, 9.0)

    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24 if PREVIEW else 128
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = (960, 540) if PREVIEW else (1600, 900)
    scene.view_settings.look = "AgX - Punchy"

    views = [
        ("front34", (4.6, -6.2, 1.55), (0, -0.25, 0.55), 50),
        ("side", (9.2, 0.0, 0.85), (0, 0.0, 0.6), 50),
        ("rear34", (-4.8, 6.0, 2.1), (0, 0.4, 0.55), 50),
        ("top34", (-5.5, -5.5, 5.2), (0, 0.0, 0.4), 50),
    ]
    if PREVIEW:
        views = views[:2]
    for name, loc, target, lens in views:
        cam.location = loc
        cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        cam_data.lens = lens
        scene.render.filepath = os.path.join(OUT, f"render_{name}.png")
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath)

# ---- Roblox FBX (each MeshPart under the triangle limit) ---------------------
for o in CAR:
    n = tris(o)
    if n > ROBLOX_TRI_LIMIT:
        dec = o.modifiers.new("dec", "DECIMATE")
        dec.ratio = ROBLOX_TRI_LIMIT / n * 0.97
        bake(o)
        print(f"  decimated {o.name}: {n} -> {tris(o)}")
select_only(CAR)
bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, "supercar_roblox.fbx"), use_selection=True,
                         apply_scale_options="FBX_SCALE_ALL", mesh_smooth_type="FACE")
print("done ->", OUT)
