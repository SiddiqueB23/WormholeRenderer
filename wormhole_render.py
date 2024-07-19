import taichi as ti

ti.init(arch=ti.cuda)

img1 = ti.Vector.field(3,dtype=ti.u8,shape=(2048,1024))
img2 = ti.Vector.field(3,dtype=ti.u8,shape=(2048,1024))
img1.from_numpy(ti.tools.image.imread('InterstellarWormhole_Fig6a.jpg'))
img2.from_numpy(ti.tools.image.imread('InterstellarWormhole_Fig10.jpg'))

WIDTH,HEIGHT = 1280,720
output_img = ti.Vector.field(3,dtype=ti.u8,shape=(WIDTH,HEIGHT))
window = ti.ui.Window("Wormhole Render", (WIDTH, HEIGHT))
canvas = window.get_canvas()
gui = window.get_gui()

num_iterations = 1000
dt = 0.01

@ti.kernel
def render(lc:ti.f32,thetac:ti.f32,phic:ti.f32,rho:ti.f32,a:ti.f32,M:ti.f32):
    for i,j in output_img:
        u = (ti.cast(i,ti.f32)-640.0)/360.0
        v = (ti.cast(j,ti.f32)-360.0)/360.0
        
        l,theta,phi = lc,thetac,phic
        thetacs = ti.math.pi/2 + ti.atan2(v,1)
        phics = 0.0 + ti.atan2(u,1)
        N = ti.math.vec3(ti.sin(thetacs)*ti.cos(phics),ti.sin(thetacs)*ti.sin(phics),ti.cos(thetacs))
        outside_wormhole = (ti.abs(l)>a)
        x = 2.0 * (ti.abs(l) - a) / (ti.math.pi * M)
        r = rho + outside_wormhole * (M * (x * ti.atan2(x,1.0) - 0.5 * ti.log(1.0 + x * x)))
        p = -N
        p.z *= -r*ti.sin(theta)
        p.y *= r
        pl,ptheta,pphi = p.x,p.y,p.z
        
        for k in range(num_iterations):
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
        
        phi = phi%(2.0*ti.math.pi)
        theta = theta%ti.math.pi
        img_u = ti.cast(phi*2048.0/(2.0*ti.math.pi),ti.u8)
        img_v = ti.cast(theta*1024.0/ti.math.pi,ti.u8)
        if(l<0):
            output_img[i,j] = img1[img_u,img_v]
        else:
            output_img[i,j] = img2[img_u,img_v]

# Parameters
lc,thetac,phic = 3.0,ti.math.pi/2,ti.math.pi/2
rho = 1.0
a = 2.0
M = 0.01
# vfov = 90

while window.running:
    window.GUI.begin("Display Panel", 0.05, 0.1, 0.2, 0.15)
    lc = gui.slider_float("lc", lc, minimum=-10, maximum=10)
    thetac = gui.slider_float("thetac", thetac, minimum=0, maximum=ti.math.pi)
    phic = gui.slider_float("phic", phic, minimum=0, maximum=2*ti.math.pi)
    rho = gui.slider_float("rho", rho, minimum=0, maximum=10)
    a = gui.slider_float("a", a, minimum=0, maximum=3)
    M = gui.slider_float("M", M, minimum=0, maximum=2)
    window.GUI.end()
    render(lc,thetac,phic,rho,a,M)
    canvas.set_image(output_img)
    window.show()