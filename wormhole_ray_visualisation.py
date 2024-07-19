import taichi as ti

ti.init(arch=ti.cuda)

window = ti.ui.Window("Wormhole Rays", (1280, 720))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(5, 2, 2)

axes = ti.Vector.field(3,dtype = ti.f32,shape = 6)
per_vertex_color_axes = ti.Vector.field(3, ti.f32, shape=6)

for i in range(0,6,2):
    axes[i][i//2] = 1.0
for i in range(3):
    per_vertex_color_axes[2*i][i] = 1.0
    per_vertex_color_axes[2*i+1][i] = 1.0

rho = 0.1
a = 0.1
M = 0.01
num_rays_w,num_rays_h = 10,10
num_rays = int(num_rays_w*num_rays_w)
num_iterations = 100
vfov = 90
lc = 10.0
phic = 0
thetac = ti.math.pi/2
ray_o = ti.math.vec3(lc,thetac,phic)
rays = ti.Vector.field(3,dtype=ti.f32,shape = (num_iterations,2,num_rays))
ray_vertices = ti.Vector.field(3,dtype=ti.f32,shape = (num_iterations*num_rays*2))
ray_vertices_colors = ti.Vector.field(3,dtype=ti.f32,shape = (num_iterations*num_rays*2))
dt = 0.1

@ti.kernel
def init_rays():
    for i in range(num_rays):
        rays[0,0,i] = ray_o
        l = rays[0,0,i].x
        u = float(i%num_rays_w)
        v = float(i/num_rays_w)
        u -= num_rays_w/2
        v -= num_rays_h/2
        u /= num_rays_h
        v /= num_rays_h
        u *= ti.tan(ti.math.radians(vfov/2))
        v *= ti.tan(ti.math.radians(vfov/2))
        thetacs = ti.math.pi/2 + ti.atan2(v,1)
        phics = ti.math.pi + ti.atan2(u,1)
        N = ti.math.vec3(ti.sin(thetacs)*ti.cos(phics),ti.sin(thetacs)*ti.sin(phics),ti.cos(thetacs))
        outside_wormhole = (ti.abs(l)>a)
        x = 2.0 * (ti.abs(l) - a) / (ti.math.pi * M)
        r = rho + outside_wormhole * (M * (x * ti.atan2(x,1.0) - 0.5 * ti.log(1.0 + x * x)))
        n = -N
        n.z *= -r*ti.sin(rays[0,0,i].y)
        n.y *= r
        rays[0,1,i] = n

@ti.kernel
def ray_integrate():
    for i in range(num_rays):
        ti.loop_config(serialize=True)
        for j in range(num_iterations-1):
            pt = rays[j,0,i]
            l,theta,phi = pt.x,pt.y,pt.z
            p = rays[j,1,i]
            pl,ptheta,pphi = p.x,p.y,p.z
            outside_wormhole = (ti.abs(l)>a)
            x = 2.0 * (ti.abs(l) - a) / (ti.math.pi * M)
            r = rho + outside_wormhole * (M * (x * ti.atan2(x,1.0) - 0.5 * ti.log(1.0 + x * x)))
            drdl = outside_wormhole * (ti.atan2(x,1.0) * 2.0 * l / (ti.math.pi * ti.abs(l)))
            b = pphi
            B = ptheta*ptheta + (pphi*pphi)/(ti.sin(theta)*ti.sin(theta))
            # A.7
            l += pl*dt
            theta += dt*ptheta/(r*r)
            phi += dt*b/(r*r*ti.sin(theta)*ti.sin(theta))
            pl += dt*B*B*drdl/(r*r*r)
            ptheta += dt*b*b*ti.cos(theta)/(r*r*ti.sin(theta)*ti.sin(theta)*ti.sin(theta))
            pt = ti.math.vec3(l,theta,phi)
            p = ti.math.vec3(pl,ptheta,pphi)
            rays[j+1,0,i] = pt
            rays[j+1,1,i] = p

@ti.kernel
def ray_to_vertices():
    for i in range(num_rays):
        ti.loop_config(serialize=False)
        for j in range(num_iterations):
            pt = rays[j,0,i]
            l,theta,phi = pt.x,pt.y,pt.z
            p = rays[j,1,i]
            pl,ptheta,pphi = p.x,p.y,p.z
            outside_wormhole = (ti.abs(l)>a)
            x = 2.0 * (ti.abs(l) - a) / (ti.math.pi * M)
            r = rho + outside_wormhole * (M * (x * ti.atan2(x,1.0) - 0.5 * ti.log(1.0 + x * x)))
            b = pphi
            ray_vertices[2*j*num_rays + 2*i] = l*ti.math.vec3(ti.sin(theta)*ti.cos(phi),ti.sin(theta)*ti.sin(phi),ti.cos(theta))
            if(l>0):
                ray_vertices_colors[2*j*num_rays + 2*i] = ti.math.vec3(1.0,1.0,0.0)
            else:
                ray_vertices_colors[2*j*num_rays + 2*i] = ti.math.vec3(1.0,0.0,1.0)
            l += pl*dt
            theta += dt*ptheta/(r*r)
            phi += dt*b/(r*r*ti.sin(theta)*ti.sin(theta))
            ray_vertices[2*j*num_rays + 2*i + 1] = l*ti.math.vec3(ti.sin(theta)*ti.cos(phi),ti.sin(theta)*ti.sin(phi),ti.cos(theta))
            if(l>0):
                ray_vertices_colors[2*j*num_rays + 2*i+1] = ti.math.vec3(1.0,1.0,0.0)
            else:
                ray_vertices_colors[2*j*num_rays + 2*i+1] = ti.math.vec3(1.0,0.0,1.0)

init_rays()
ray_integrate()
ray_to_vertices()

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((1.0, 1.0, 1.0))
    scene.lines(axes,width = 1.0,per_vertex_color=per_vertex_color_axes)
    
    # init_rays()
    # ray_integrate()
    # ray_to_vertices()
    scene.lines(ray_vertices,width = 1.0,per_vertex_color=ray_vertices_colors)
    canvas.scene(scene)
    window.show()
