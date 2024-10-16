# -*- coding:utf-8 -*-

from odbAccess import openOdb
from textRepr import *
from abaqus import*
from abaqusConstants import*
from caeModules import *
import csv
import regionToolset
import time
import os
from driverUtils import executeOnCaeStartup

session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=317.6875, height=205.333343505859)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()

executeOnCaeStartup()
openMdb('test.cae')
session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=ON)
session.viewports['Viewport: 1'].setValues(displayedObject=None)
time1 = time.time()
# Filename = ('D:/temp/tpms/test1/gradient_class1_'+str(i)+'.inp')
Filename = ('D:/temp/tpms/shoe tie/0.08_0.12_0.16/0.08_0.12_0.16.inp')
if os.path.exists(Filename):
    start_time = time.time()
    mdb.ModelFromInputFile(name='test', inputFileName=Filename)
    # define the properties of materials
    mdb.models['test'].Material(name='tpu-wenext')
    mdb.models['test'].materials['tpu-wenext'].Density(table=((1e-09, ), ))
    mdb.models['test'].materials['tpu-wenext'].Hyperelastic(materialType=ISOTROPIC, 
        type=POLYNOMIAL, n=2, volumetricResponse=POISSON_RATIO, poissonRatio=0.47, 
        table=())
    mdb.models['test'].materials['tpu-wenext'].hyperelastic.UniaxialTestData(
        table=((-0.0096, 0.0), (0.0024, 0.0002), (0.0174, 0.0011), (0.03264, 
        0.002), (0.04968, 0.0028), (0.06756, 0.0037), (0.08628, 0.0045), (0.10524, 
        0.0053), (0.12456, 0.0062), (0.14388, 0.007), (0.16332, 0.0078), (0.18252, 
        0.0087), (0.20172, 0.0095), (0.2208, 0.0103), (0.23964, 0.0112), (0.25836, 
        0.012), (0.27696, 0.0128), (0.29532, 0.0137), (0.31356, 0.0145), (0.33156, 
        0.0153), (0.34932, 0.0162), (0.36696, 0.017), (0.38436, 0.0178), (0.40164, 
        0.0187), (0.41856, 0.0195), (0.43536, 0.0203), (0.45192, 0.0212), (0.46836, 
        0.022), (0.48444, 0.0228), (0.50052, 0.0237), (0.51624, 0.0245), (0.53184, 
        0.0253), (0.5472, 0.0262), (0.56232, 0.027), (0.5772, 0.0278), (0.59208, 
        0.0287), (0.60672, 0.0295), (0.62112, 0.0303), (0.6354, 0.0312), (0.64956, 
        0.032), (0.6636, 0.0328), (0.67752, 0.0337), (0.69132, 0.0345), (0.70488, 
        0.0353), (0.71844, 0.0362), (0.73176, 0.037), (0.74496, 0.0378), (0.75816, 
        0.0387), (0.77112, 0.0395), (0.78408, 0.0403), (0.7968, 0.0412), (0.8094, 
        0.042), (0.822, 0.0428), (0.83436, 0.0437), (0.8466, 0.0445), (0.85884, 
        0.0453), (0.87096, 0.0462), (0.88308, 0.047), (0.89496, 0.0478), (0.90684, 
        0.0487), (0.9186, 0.0495), (0.93024, 0.0503), (0.94164, 0.0512), (0.95304, 
        0.052), (0.96432, 0.0528), (0.9756, 0.0537), (0.98664, 0.0545), (0.99768, 
        0.0553), (1.00872, 0.0562), (1.01964, 0.057), (1.03044, 0.0578), (1.04112, 
        0.0587), (1.0518, 0.0595), (1.06236, 0.0603), (1.07292, 0.0612), (1.08336, 
        0.062), (1.0938, 0.0628), (1.10412, 0.0637), (1.11432, 0.0645), (1.12452, 
        0.0653), (1.13472, 0.0662), (1.14492, 0.067), (1.155, 0.0678), (1.16508, 
        0.0687), (1.17516, 0.0695), (1.185, 0.0703), (1.19496, 0.0712), (1.20468, 
        0.072), (1.21452, 0.0728), (1.22424, 0.0737), (1.23384, 0.0745), (1.24356, 
        0.0753), (1.25328, 0.0762), (1.26288, 0.077), (1.27248, 0.0778), (1.28196, 
        0.0787), (1.29132, 0.0795), (1.30056, 0.0803), (1.30968, 0.0812), (1.3188, 
        0.082), (1.3278, 0.0828), (1.33668, 0.0837), (1.34544, 0.0845), (1.35432, 
        0.0853), (1.36308, 0.0861), (1.37172, 0.087), (1.38036, 0.0878), (1.389, 
        0.0887), (1.39752, 0.0895), (1.40604, 0.0903), (1.41444, 0.0912), (1.42284, 
        0.092), (1.43124, 0.0928), (1.43952, 0.0937), (1.4478, 0.0945), (1.45596, 
        0.0953), (1.46412, 0.0962), (1.47228, 0.097), (1.48044, 0.0978), (1.48848, 
        0.0987), (1.49652, 0.0995), (1.50444, 0.1003), (1.51236, 0.1012), (1.52004, 
        0.102), (1.52784, 0.1028), (1.53552, 0.1037), (1.5432, 0.1045), (1.55088, 
        0.1053), (1.55844, 0.1062), (1.566, 0.107), (1.57356, 0.1078), (1.58112, 
        0.1087), (1.58868, 0.1095), (1.59612, 0.1103), (1.60356, 0.1112), (1.611, 
        0.112), (1.61832, 0.1128), (1.62552, 0.1137), (1.6326, 0.1145), (1.6398, 
        0.1153), (1.64676, 0.1162), (1.65396, 0.117), (1.66104, 0.1178), (1.66812, 
        0.1187), (1.67508, 0.1195), (1.68216, 0.1203), (1.68912, 0.1212), (1.69608, 
        0.122), (1.70292, 0.1228), (1.70964, 0.1236), (1.71636, 0.1245), (1.72308, 
        0.1253), (1.7298, 0.1262), (1.7364, 0.127), (1.74312, 0.1278), (1.74984, 
        0.1287), (1.75644, 0.1295), (1.76316, 0.1303), (1.76988, 0.1312), (1.77648, 
        0.132), (1.78308, 0.1328), (1.78956, 0.1337), (1.79616, 0.1345), (1.80252, 
        0.1353), (1.80888, 0.1362), (1.81524, 0.137), (1.82148, 0.1378), (1.82772, 
        0.1387), (1.83384, 0.1395), (1.83996, 0.1403), (1.84608, 0.1412), (1.8522, 
        0.142), (1.85832, 0.1428), (1.86444, 0.1437), (1.87056, 0.1445), (1.87668, 
        0.1453), (1.88268, 0.1462), (1.88868, 0.147), (1.89468, 0.1478), (1.90056, 
        0.1487), (1.90644, 0.1495), (1.91232, 0.1503), (1.91808, 0.1512), (1.92396, 
        0.152), (1.92972, 0.1528), (1.93548, 0.1537), (1.94124, 0.1545), (1.947, 
        0.1553), (1.95276, 0.1562), (1.9584, 0.157), (1.96404, 0.1578), (1.96968, 
        0.1587), (1.9752, 0.1595), (1.98072, 0.1603), (1.98624, 0.1612), (1.99176, 
        0.162), (1.99716, 0.1628), (2.00256, 0.1637), (2.00796, 0.1645), (2.01324, 
        0.1653), (2.01864, 0.1662), (2.02392, 0.167), (2.0292, 0.1678), (2.03448, 
        0.1687), (2.03976, 0.1695), (2.04504, 0.1703), (2.0502, 0.1712), (2.05536, 
        0.172), (2.06052, 0.1728), (2.06556, 0.1737), (2.07072, 0.1745), (2.07576, 
        0.1753), (2.0808, 0.1762), (2.08584, 0.177), (2.09088, 0.1778), (2.09592, 
        0.1787), (2.10084, 0.1795), (2.10588, 0.1803), (2.11092, 0.1812), (2.11584, 
        0.182), (2.12076, 0.1828), (2.12568, 0.1837), (2.13072, 0.1845), (2.13564, 
        0.1853), (2.14044, 0.1862), (2.14536, 0.187), (2.15016, 0.1878), (2.15496, 
        0.1887), (2.15976, 0.1895), (2.16456, 0.1903), (2.16924, 0.1912), (2.17404, 
        0.192), (2.17884, 0.1928), (2.18352, 0.1937), (2.18832, 0.1945), (2.193, 
        0.1953), (2.19756, 0.1962), (2.20212, 0.197), (2.2068, 0.1978), (2.21136, 
        0.1986), (2.21604, 0.1995), (2.2206, 0.2003), (2.22516, 0.2012), (2.22972, 
        0.202), (2.23416, 0.2028), (2.2386, 0.2037), (2.24292, 0.2045), (2.24736, 
        0.2053), (2.25168, 0.2062), (2.256, 0.207), (2.26032, 0.2078), (2.26464, 
        0.2087), (2.26896, 0.2095), (2.27328, 0.2103), (2.27772, 0.2112), (2.28204, 
        0.212), (2.28636, 0.2128), (2.29056, 0.2137), (2.29488, 0.2145), (2.29908, 
        0.2153), (2.30328, 0.2162), (2.30748, 0.217), (2.31168, 0.2178), (2.31588, 
        0.2187), (2.32008, 0.2195), (2.32416, 0.2203), (2.32836, 0.2212), (2.33244, 
        0.222), (2.33652, 0.2228), (2.3406, 0.2237), (2.34468, 0.2245), (2.34864, 
        0.2253), (2.35272, 0.2262), (2.3568, 0.227), (2.36088, 0.2278), (2.36496, 
        0.2287), (2.36904, 0.2295), (2.373, 0.2303), (2.37708, 0.2312), (2.38104, 
        0.232), (2.385, 0.2328), (2.38896, 0.2337), (2.3928, 0.2345), (2.39664, 
        0.2353), (2.40048, 0.2361), (2.40432, 0.237), (2.40816, 0.2378), (2.412, 
        0.2387), (2.41584, 0.2395), (2.41956, 0.2403), (2.4234, 0.2412), (2.42724, 
        0.242), (2.43096, 0.2428), (2.43456, 0.2437), (2.43828, 0.2445), (2.44188, 
        0.2453), (2.44548, 0.2462), (2.44908, 0.247), (2.45268, 0.2478), (2.4564, 
        0.2487), (2.46012, 0.2495), (2.46384, 0.2503), (2.46756, 0.2512), (2.47128, 
        0.252), (2.475, 0.2528), (2.47872, 0.2537), (2.48244, 0.2545), (2.48604, 
        0.2553), (2.48976, 0.2562), (2.49336, 0.257), (2.49696, 0.2578), (2.50068, 
        0.2587), (2.50428, 0.2595), (2.50788, 0.2603), (2.51148, 0.2612), (2.51508, 
        0.262), (2.51868, 0.2628), (2.52216, 0.2637), (2.52576, 0.2645), (2.52924, 
        0.2653), (2.53272, 0.2662), (2.5362, 0.267), (2.53968, 0.2678), (2.54316, 
        0.2687), (2.54664, 0.2695), (2.55012, 0.2703), (2.55348, 0.2712), (2.55696, 
        0.272), (2.56032, 0.2728), (2.56368, 0.2737), (2.56704, 0.2745), (2.5704, 
        0.2753), (2.57376, 0.2762), (2.57712, 0.277), (2.5806, 0.2778), (2.58396, 
        0.2787), (2.58744, 0.2795), (2.5908, 0.2803), (2.59416, 0.2812), (2.59752, 
        0.282), (2.60076, 0.2828), (2.60412, 0.2837), (2.60736, 0.2845), (2.61072, 
        0.2853), (2.61408, 0.2862), (2.61732, 0.287), (2.62056, 0.2878), (2.62368, 
        0.2887), (2.62692, 0.2895), (2.63004, 0.2903), (2.63316, 0.2912), (2.6364, 
        0.292), (2.63964, 0.2928), (2.64288, 0.2937), (2.64612, 0.2945), (2.64936, 
        0.2953), (2.6526, 0.2962), (2.65572, 0.297), (2.65872, 0.2978), (2.66184, 
        0.2987), (2.66496, 0.2995), (2.66796, 0.3003), (2.67108, 0.3012), (2.6742, 
        0.302), (2.67732, 0.3028), (2.68032, 0.3037), (2.68344, 0.3045), (2.68656, 
        0.3053), (2.68956, 0.3062), (2.69268, 0.307), (2.69568, 0.3078), (2.6988, 
        0.3087), (2.7018, 0.3095), (2.7048, 0.3103), (2.7078, 0.3111), (2.7108, 
        0.312), (2.7138, 0.3128), (2.7168, 0.3137), (2.7198, 0.3145), (2.72268, 
        0.3153), (2.72568, 0.3162), (2.72856, 0.317), (2.73144, 0.3178), (2.73432, 
        0.3187), (2.73732, 0.3195), (2.7402, 0.3203), (2.74308, 0.3212), (2.74608, 
        0.322), (2.74896, 0.3228), (2.75196, 0.3237), (2.75472, 0.3245), (2.7576, 
        0.3253), (2.76036, 0.3262), (2.76324, 0.327), (2.76612, 0.3278), (2.769, 
        0.3287), (2.77188, 0.3295), (2.77476, 0.3303), (2.77764, 0.3312), (2.78052, 
        0.332), (2.78328, 0.3328), (2.78616, 0.3337), (2.78892, 0.3345), (2.79168, 
        0.3353), (2.79444, 0.3362), (2.7972, 0.337), (2.79996, 0.3378), (2.80272, 
        0.3387), (2.80548, 0.3395), (2.80824, 0.3403), (2.811, 0.3412), (2.81364, 
        0.342), (2.8164, 0.3428), (2.81904, 0.3437), (2.82168, 0.3445), (2.82444, 
        0.3453), (2.82708, 0.3462), (2.82972, 0.347), (2.83236, 0.3478), (2.83512, 
        0.3487), (2.83776, 0.3495), (2.8404, 0.3503), (2.84304, 0.3512), (2.84568, 
        0.352), (2.84832, 0.3528), (2.85096, 0.3537), (2.8536, 0.3545), (2.85624, 
        0.3553), (2.85876, 0.3562), (2.8614, 0.357), (2.86404, 0.3578), (2.86668, 
        0.3587), (2.86932, 0.3595), (2.87196, 0.3603), (2.87448, 0.3612), (2.877, 
        0.362), (2.87964, 0.3628), (2.88204, 0.3637), (2.88456, 0.3645), (2.88708, 
        0.3653), (2.88948, 0.3662), (2.89188, 0.367), (2.8944, 0.3678), (2.8968, 
        0.3687), (2.8992, 0.3695), (2.9016, 0.3703), (2.90412, 0.3712), (2.90652, 
        0.372), (2.90904, 0.3728), (2.91144, 0.3737), (2.91384, 0.3745), (2.91624, 
        0.3753), (2.91864, 0.3762), (2.92104, 0.377), (2.92344, 0.3778), (2.92584, 
        0.3787), (2.92824, 0.3795), (2.93064, 0.3803), (2.93304, 0.3812), (2.93556, 
        0.382), (2.93796, 0.3828), (2.94036, 0.3837), (2.94276, 0.3845), (2.94504, 
        0.3853), (2.94744, 0.3862), (2.94984, 0.387), (2.95224, 0.3878), (2.95464, 
        0.3887), (2.95704, 0.3895), (2.95944, 0.3903), (2.96184, 0.3912), (2.96412, 
        0.392), (2.96652, 0.3928), (2.9688, 0.3937), (2.97096, 0.3945), (2.97324, 
        0.3953), (2.97552, 0.3962), (2.9778, 0.397), (2.97996, 0.3978), (2.98224, 
        0.3987), (2.98464, 0.3995), (2.98692, 0.4003), (2.9892, 0.4012), (2.99148, 
        0.402), (2.99376, 0.4028), (2.99604, 0.4037), (2.99832, 0.4045), (3.00072, 
        0.4053), (3.003, 0.4062), (3.00528, 0.407), (3.00756, 0.4078), (3.00984, 
        0.4087), (3.01212, 0.4095), (3.01428, 0.4103), (3.01644, 0.4112), (3.0186, 
        0.412), (3.02088, 0.4128), (3.02304, 0.4137), (3.02532, 0.4145), (3.0276, 
        0.4153), (3.02988, 0.4162), (3.03216, 0.417), (3.03432, 0.4178), (3.0366, 
        0.4187), (3.03888, 0.4195), (3.04104, 0.4203), (3.04332, 0.4212), (3.0456, 
        0.422), (3.04776, 0.4228), (3.05004, 0.4236), (3.0522, 0.4245), (3.05448, 
        0.4253), (3.05676, 0.4262), (3.05904, 0.427), (3.06132, 0.4278), (3.06348, 
        0.4287), (3.06576, 0.4295), (3.0678, 0.4303), (3.06996, 0.4312), (3.072, 
        0.432), (3.07416, 0.4328), (3.0762, 0.4337), (3.07836, 0.4345), (3.08052, 
        0.4353), (3.08268, 0.4362), (3.08484, 0.437), (3.08688, 0.4378), (3.08892, 
        0.4387), (3.09084, 0.4395), (3.09276, 0.4403), (3.09468, 0.4412), (3.0966, 
        0.442), (3.09852, 0.4428), (3.10044, 0.4437), (3.10236, 0.4445), (3.10428, 
        0.4453), (3.1062, 0.4462), (3.10812, 0.447), (3.11004, 0.4478), (3.11196, 
        0.4487), (3.114, 0.4495), (3.11604, 0.4503), (3.11808, 0.4512), (3.12012, 
        0.452), (3.12228, 0.4528), (3.12444, 0.4537), (3.1266, 0.4545), (3.12888, 
        0.4553), (3.13104, 0.4562), (3.1332, 0.457), (3.13536, 0.4578), (3.13752, 
        0.4587), (3.13968, 0.4595), (3.14184, 0.4603), (3.14388, 0.4612), (3.14592, 
        0.462), (3.14796, 0.4628), (3.14988, 0.4637), (3.15192, 0.4645), (3.15396, 
        0.4653), (3.156, 0.4662), (3.15792, 0.467), (3.15996, 0.4678), (3.162, 
        0.4687), (3.16392, 0.4695), (3.16584, 0.4703), (3.16776, 0.4712), (3.16968, 
        0.472), (3.17172, 0.4728), (3.17376, 0.4737), (3.1758, 0.4745), (3.17772, 
        0.4753), (3.17976, 0.4762), (3.18168, 0.477), (3.18348, 0.4778), (3.1854, 
        0.4787), (3.18732, 0.4795), (3.18924, 0.4803), (3.19128, 0.4812), (3.1932, 
        0.482), (3.19524, 0.4828), (3.19728, 0.4837), (3.19932, 0.4845), (3.20124, 
        0.4853), (3.20328, 0.4862), (3.2052, 0.487), (3.20712, 0.4878), (3.20904, 
        0.4887), (3.21108, 0.4895), (3.213, 0.4903), (3.21492, 0.4912), (3.21696, 
        0.492), (3.21888, 0.4928), (3.22092, 0.4937), (3.22284, 0.4945), (3.22476, 
        0.4953), (3.2268, 0.4962), (3.22872, 0.497), (3.23052, 0.4978), (3.23244, 
        0.4987), (3.23424, 0.4995), (3.23604, 0.5003), (3.23796, 0.5012), (3.23988, 
        0.502), (3.24168, 0.5028), (3.2436, 0.5037), (3.24552, 0.5045), (3.24756, 
        0.5053), (3.24948, 0.5062), (3.2514, 0.507), (3.25332, 0.5078), (3.25524, 
        0.5087), (3.25704, 0.5095), (3.25884, 0.5103), (3.26076, 0.5112), (3.26256, 
        0.512), (3.26436, 0.5128), (3.26616, 0.5137), (3.26784, 0.5145), (3.26964, 
        0.5153), (3.27144, 0.5162), (3.27324, 0.517), (3.27504, 0.5178), (3.27684, 
        0.5187), (3.27864, 0.5195), (3.28044, 0.5203), (3.28224, 0.5212), (3.28404, 
        0.522), (3.28584, 0.5228), (3.28764, 0.5237), (3.28956, 0.5245), (3.29136, 
        0.5253), (3.29316, 0.5262), (3.29484, 0.527), (3.29664, 0.5278), (3.29832, 
        0.5287), (3.3, 0.5295), (3.30168, 0.5303), (3.30336, 0.5312), (3.30504, 
        0.532), (3.30672, 0.5328), (3.30828, 0.5337), (3.30996, 0.5345), (3.31152, 
        0.5353), (3.3132, 0.5361), (3.315, 0.537), (3.31668, 0.5378), (3.31848, 
        0.5387), (3.32028, 0.5395), (3.32208, 0.5403), (3.32376, 0.5412), (3.32544, 
        0.542), (3.32712, 0.5428), (3.32868, 0.5437), (3.33024, 0.5445), (3.33192, 
        0.5453), (3.33348, 0.5462), (3.33516, 0.547), (3.33684, 0.5478), (3.33864, 
        0.5487), (3.34032, 0.5495), (3.342, 0.5503), (3.34356, 0.5512), (3.34524, 
        0.552), (3.3468, 0.5528), (3.34836, 0.5537), (3.34992, 0.5545), (3.3516, 
        0.5553), (3.35316, 0.5562), (3.35484, 0.557), (3.35664, 0.5578), (3.35832, 
        0.5587), (3.36, 0.5595), (3.3618, 0.5603), (3.36348, 0.5612), (3.36504, 
        0.562), (3.36672, 0.5628), (3.36828, 0.5637), (3.36996, 0.5645), (3.37152, 
        0.5653), (3.37308, 0.5662), (3.37476, 0.567), (3.37632, 0.5678), (3.37788, 
        0.5687), (3.37944, 0.5695), (3.381, 0.5703), (3.38268, 0.5712), (3.38436, 
        0.572), (3.38604, 0.5728), (3.38772, 0.5736), (3.3894, 0.5745), (3.39108, 
        0.5753), (3.39276, 0.5762), (3.39444, 0.577), (3.396, 0.5778), (3.39768, 
        0.5787), (3.39936, 0.5795), (3.40092, 0.5803), (3.4026, 0.5812), (3.40428, 
        0.582), (3.40596, 0.5828), (3.40764, 0.5837), (3.40944, 0.5845), (3.41112, 
        0.5853), (3.4128, 0.5862), (3.41436, 0.587), (3.41604, 0.5878), (3.4176, 
        0.5887), (3.41916, 0.5895), (3.42072, 0.5903), (3.42228, 0.5912), (3.42396, 
        0.592), (3.42552, 0.5928), (3.42696, 0.5937), (3.42852, 0.5945), (3.43008, 
        0.5953), (3.43152, 0.5962), (3.43308, 0.597), (3.43464, 0.5978), (3.43608, 
        0.5987), (3.43752, 0.5995), (3.43908, 0.6003), (3.44052, 0.6012), (3.44196, 
        0.602), (3.44352, 0.6028), (3.44496, 0.6037), (3.4464, 0.6045), (3.44784, 
        0.6053), (3.44928, 0.6062), (3.45072, 0.607), (3.45228, 0.6078), (3.45372, 
        0.6087), (3.45516, 0.6095), (3.45672, 0.6103), (3.45816, 0.6111), (3.4596, 
        0.612), (3.46104, 0.6128), (3.4626, 0.6137), (3.46404, 0.6145), (3.46548, 
        0.6153), (3.46704, 0.6162), (3.46848, 0.617), (3.46992, 0.6178), (3.47136, 
        0.6187), (3.4728, 0.6195), (3.47424, 0.6203), (3.47568, 0.6212), (3.47712, 
        0.622), (3.47868, 0.6228), (3.48012, 0.6237), (3.48156, 0.6245), (3.483, 
        0.6253), (3.48444, 0.6262), (3.48576, 0.627), (3.4872, 0.6278), (3.48864, 
        0.6287), (3.4902, 0.6295), (3.49176, 0.6303), (3.49332, 0.6312), (3.49488, 
        0.632), (3.49632, 0.6328), (3.49788, 0.6337), (3.49944, 0.6345), (3.501, 
        0.6353), (3.50256, 0.6362), (3.50424, 0.637), (3.50592, 0.6378), (3.5076, 
        0.6387), (3.50928, 0.6395), (3.51084, 0.6403), (3.51252, 0.6412), (3.51408, 
        0.642), (3.51552, 0.6428), (3.51708, 0.6437), (3.51852, 0.6445), (3.51996, 
        0.6453), (3.52152, 0.6462), (3.52296, 0.647), (3.52452, 0.6478), (3.52596, 
        0.6487), (3.52752, 0.6495), (3.52896, 0.6503), (3.5304, 0.6512), (3.53184, 
        0.652), (3.5334, 0.6528), (3.53484, 0.6537), (3.5364, 0.6545), (3.53784, 
        0.6553), (3.5394, 0.6562), (3.54084, 0.657), (3.54228, 0.6578), (3.54372, 
        0.6587), (3.54516, 0.6595), (3.54672, 0.6603), (3.54828, 0.6612), (3.54972, 
        0.662), (3.55128, 0.6628), (3.55272, 0.6637), (3.55416, 0.6645), (3.55548, 
        0.6653), (3.55692, 0.6662), (3.55824, 0.667), (3.55968, 0.6678), (3.56112, 
        0.6687), (3.56256, 0.6695), (3.56412, 0.6703), (3.56556, 0.6712), (3.56712, 
        0.672), (3.56856, 0.6728), (3.57, 0.6737), (3.57132, 0.6745), (3.57276, 
        0.6753), (3.5742, 0.6762), (3.57564, 0.677), (3.57708, 0.6778), (3.57852, 
        0.6787), (3.57996, 0.6795), (3.5814, 0.6803), (3.58296, 0.6812), (3.5844, 
        0.682), (3.58584, 0.6828), (3.5874, 0.6837), (3.58884, 0.6845), (3.59028, 
        0.6853), (3.59172, 0.6862), (3.59316, 0.687), (3.5946, 0.6878), (3.59604, 
        0.6887), (3.5976, 0.6895), (3.59904, 0.6903), (3.6006, 0.6912), (3.60204, 
        0.692), (3.6036, 0.6928), (3.60516, 0.6937), (3.6066, 0.6945), (3.60816, 
        0.6953), (3.6096, 0.6962), (3.61104, 0.697), (3.61248, 0.6978), (3.61392, 
        0.6987), (3.61548, 0.6995), (3.61692, 0.7003), (3.61836, 0.7012), (3.61992, 
        0.702), (3.62136, 0.7028), (3.62292, 0.7037), (3.62436, 0.7045), (3.62592, 
        0.7053), (3.62736, 0.7062), (3.6288, 0.707), (3.63024, 0.7078), (3.63168, 
        0.7087), (3.63312, 0.7095), (3.63444, 0.7103), (3.63588, 0.7112), (3.6372, 
        0.712), (3.63852, 0.7128), (3.63984, 0.7137), (3.64116, 0.7145), (3.6426, 
        0.7153), (3.64392, 0.7162), (3.64524, 0.717), (3.64668, 0.7178), (3.648, 
        0.7187), (3.64944, 0.7195), (3.65076, 0.7203), (3.65208, 0.7212), (3.6534, 
        0.722), (3.65472, 0.7228), (3.65604, 0.7236), (3.65736, 0.7245), (3.65868, 
        0.7253), (3.66, 0.7262), (3.6612, 0.727), (3.66252, 0.7278), (3.66372, 
        0.7287), (3.66504, 0.7295), (3.66636, 0.7303), (3.66768, 0.7312), (3.669, 
        0.732), (3.67032, 0.7328), (3.67176, 0.7337), (3.67308, 0.7345), (3.67428, 
        0.7353), (3.6756, 0.7362), (3.6768, 0.737), (3.678, 0.7378), (3.6792, 
        0.7387), (3.68052, 0.7395), (3.68172, 0.7403), (3.68292, 0.7412), (3.68412, 
        0.742), (3.68532, 0.7428), (3.68652, 0.7437), (3.68784, 0.7445), (3.68916, 
        0.7453), (3.6906, 0.7462), (3.69192, 0.747), (3.69336, 0.7478), (3.69468, 
        0.7487), (3.69588, 0.7495), (3.69708, 0.7503), (3.69828, 0.7512), (3.69948, 
        0.752), (3.70068, 0.7528), (3.702, 0.7537), (3.70344, 0.7545), (3.70476, 
        0.7553), (3.70608, 0.7562), (3.7074, 0.757), (3.70872, 0.7578), (3.70992, 
        0.7587), (3.71112, 0.7595), (3.71232, 0.7603), (3.7134, 0.7611), (3.7146, 
        0.762), (3.7158, 0.7628), (3.717, 0.7637), (3.7182, 0.7645), (3.71952, 
        0.7653), (3.72084, 0.7662), (3.72216, 0.767), (3.72348, 0.7678), (3.7248, 
        0.7687), (3.72612, 0.7695), (3.72744, 0.7703), (3.72864, 0.7712), (3.72996, 
        0.772), (3.73128, 0.7728), (3.73248, 0.7737), (3.7338, 0.7745), (3.73512, 
        0.7753), (3.73656, 0.7762), (3.738, 0.777), (3.73956, 0.7778), (3.741, 
        0.7787), (3.74256, 0.7795), (3.744, 0.7803), (3.74544, 0.7812), (3.74688, 
        0.782), (3.7482, 0.7828), (3.74964, 0.7837), (3.75096, 0.7845), (3.7524, 
        0.7853), (3.75372, 0.7862), (3.75504, 0.787), (3.75636, 0.7878), (3.7578, 
        0.7887), (3.75912, 0.7895), (3.76056, 0.7903), (3.762, 0.7912), (3.76332, 
        0.792), (3.76476, 0.7928), (3.76608, 0.7937), (3.7674, 0.7945), (3.76872, 
        0.7953), (3.77004, 0.7962), (3.77136, 0.797), (3.77268, 0.7978), (3.77388, 
        0.7987), (3.77508, 0.7995), (3.77628, 0.8003), (3.77748, 0.8012), (3.77868, 
        0.802), (3.77988, 0.8028), (3.78096, 0.8037), (3.78216, 0.8045), (3.78336, 
        0.8053), (3.78456, 0.8062), (3.78588, 0.807), (3.78708, 0.8078), (3.78828, 
        0.8087), (3.78948, 0.8095), (3.79068, 0.8103), (3.79176, 0.8112), (3.79296, 
        0.812), (3.79404, 0.8128), (3.79524, 0.8137), (3.79632, 0.8145), (3.7974, 
        0.8153), (3.7986, 0.8162), (3.7998, 0.817), (3.801, 0.8178), (3.8022, 
        0.8187), (3.8034, 0.8195), (3.80448, 0.8203), (3.80556, 0.8212), (3.80664, 
        0.822), (3.80772, 0.8228), (3.80892, 0.8237), (3.81, 0.8245), (3.8112, 
        0.8253), (3.81252, 0.8262), (3.81372, 0.827), (3.8148, 0.8278), (3.81588, 
        0.8287), (3.81708, 0.8295), (3.81804, 0.8303), (3.81924, 0.8312), (3.82032, 
        0.832), (3.82152, 0.8328), (3.82272, 0.8337), (3.82392, 0.8345), (3.82512, 
        0.8353), (3.82632, 0.8361), (3.82752, 0.837), (3.82872, 0.8378), (3.83004, 
        0.8387), (3.83136, 0.8395), (3.83256, 0.8403), (3.83376, 0.8412), (3.83508, 
        0.842), (3.83616, 0.8428), (3.83736, 0.8437), (3.83844, 0.8445), (3.83964, 
        0.8453), (3.84072, 0.8462), (3.84192, 0.847), (3.84312, 0.8478), (3.8442, 
        0.8487), (3.8454, 0.8495), (3.84648, 0.8503), (3.84756, 0.8512), (3.84864, 
        0.852), (3.84972, 0.8528), (3.8508, 0.8537), (3.85188, 0.8545), (3.85296, 
        0.8553), (3.85404, 0.8562), (3.85512, 0.857), (3.85608, 0.8578), (3.85716, 
        0.8587), (3.85824, 0.8595), (3.8592, 0.8603), (3.86028, 0.8612), (3.86136, 
        0.862), (3.86244, 0.8628), (3.86352, 0.8637), (3.8646, 0.8645), (3.86568, 
        0.8653), (3.86676, 0.8662), (3.86784, 0.867), (3.8688, 0.8678), (3.86976, 
        0.8687), (3.87072, 0.8695), (3.87156, 0.8703), (3.8724, 0.8712), (3.87336, 
        0.872), (3.87432, 0.8728), (3.87528, 0.8737), (3.87636, 0.8745), (3.87744, 
        0.8753), (3.87864, 0.8762), (3.87972, 0.877), (3.8808, 0.8778), (3.88188, 
        0.8787), (3.88296, 0.8795), (3.88392, 0.8803), (3.88488, 0.8812), (3.88584, 
        0.882), (3.8868, 0.8828), (3.88776, 0.8837), (3.88872, 0.8845), (3.88968, 
        0.8853), (3.89076, 0.8862), (3.89172, 0.887), (3.89268, 0.8878), (3.89376, 
        0.8887), (3.89472, 0.8895), (3.8958, 0.8903), (3.89688, 0.8912), (3.89796, 
        0.892), (3.89904, 0.8928), (3.90012, 0.8937), (3.9012, 0.8945), (3.90216, 
        0.8953), (3.90324, 0.8962), (3.9042, 0.897), (3.90528, 0.8978), (3.90636, 
        0.8987), (3.90744, 0.8995), (3.90852, 0.9003), (3.90972, 0.9012), (3.9108, 
        0.902), (3.912, 0.9028), (3.9132, 0.9037), (3.91428, 0.9045), (3.91536, 
        0.9053), (3.91632, 0.9062), (3.9174, 0.907), (3.91848, 0.9078), (3.91944, 
        0.9087), (3.92052, 0.9095), (3.9216, 0.9103), (3.92268, 0.9112), (3.92376, 
        0.912), (3.92484, 0.9128), (3.92592, 0.9137), (3.927, 0.9145), (3.92796, 
        0.9153), (3.9288, 0.9162), (3.92976, 0.917), (3.9306, 0.9178), (3.93156, 
        0.9187), (3.93252, 0.9195), (3.9336, 0.9203), (3.93456, 0.9212), (3.93564, 
        0.922), (3.9366, 0.9228), (3.93768, 0.9237), (3.93864, 0.9245), (3.93972, 
        0.9253), (3.9408, 0.9262), (3.94176, 0.927), (3.94284, 0.9278), (3.94392, 
        0.9287), (3.945, 0.9295), (3.94608, 0.9303), (3.94716, 0.9312), (3.94812, 
        0.932), (3.94908, 0.9328), (3.95004, 0.9337), (3.951, 0.9345), (3.95196, 
        0.9353), (3.95292, 0.9362), (3.95388, 0.937), (3.95472, 0.9378), (3.95568, 
        0.9387), (3.95664, 0.9395), (3.9576, 0.9403), (3.95844, 0.9412), (3.9594, 
        0.942), (3.96036, 0.9428), (3.96132, 0.9437), (3.96216, 0.9445), (3.96312, 
        0.9453), (3.96396, 0.9462), (3.96492, 0.947), (3.96588, 0.9478), (3.96684, 
        0.9487), (3.9678, 0.9495), (3.96876, 0.9503), (3.96972, 0.9512), (3.97068, 
        0.952), (3.97152, 0.9528), (3.97248, 0.9537), (3.97344, 0.9545), (3.9744, 
        0.9553), (3.97536, 0.9562), (3.9762, 0.957), (3.97716, 0.9578), (3.978, 
        0.9587), (3.97884, 0.9595), (3.9798, 0.9603), (3.98064, 0.9612), (3.98136, 
        0.962), (3.9822, 0.9628), (3.98304, 0.9637), (3.98388, 0.9645), (3.98484, 
        0.9653), (3.9858, 0.9662), (3.98676, 0.967), (3.98772, 0.9678), (3.98868, 
        0.9687), (3.98952, 0.9695), (3.99048, 0.9703), (3.99132, 0.9712), (3.99216, 
        0.972), (3.993, 0.9728), (3.99384, 0.9737), (3.99468, 0.9745), (3.99552, 
        0.9753), (3.99636, 0.9762), (3.99732, 0.977), (3.99816, 0.9778), (3.999, 
        0.9787), (3.99996, 0.9795), (4.0008, 0.9803), (4.00164, 0.9812), (4.00248, 
        0.982), (4.0032, 0.9828), (4.00404, 0.9837), (4.00476, 0.9845), (4.0056, 
        0.9853), (4.00632, 0.9862), (4.00704, 0.987), (4.00788, 0.9878), (4.0086, 
        0.9887), (4.00944, 0.9895), (4.01028, 0.9903), (4.011, 0.9912), (4.01184, 
        0.992), (4.01256, 0.9928), (4.0134, 0.9937), (4.01412, 0.9945), (4.01484, 
        0.9953), (4.01568, 0.9962), (4.01628, 0.997), (4.017, 0.9978), (4.01772, 
        0.9987), (4.01844, 0.9995), (4.01904, 1.0003)))
    mdb.models['test'].materials['tpu-wenext'].Plastic(scaleStress=None, table=((
        0.78423, 0.0), (1.35487, 0.0116319), (2.15677, 0.0249853), (2.7908, 
        0.0439207), (3.37279, 0.0679816), (3.91174, 0.0920318), (4.42433, 
        0.114195), (4.92701, 0.138108), (5.44151, 0.161523), (5.92832, 0.187184), (
        6.39031, 0.215918)))
    mdb.models['test'].HomogeneousSolidSection(name='Section-1', material='tpu-wenext', thickness=None)
    # Creating sections and assigning section properties
    p = mdb.models['test'].parts['PART-1']
    e = p.elements
    elements = e.getByBoundingBox(xMin=-21.0, xMax=21.0, yMin=-21.0, yMax=21.0, zMin=-21.0, zMax=21.0)
    region = p.Set(elements=elements, name='Set-1')
    p = mdb.models['test'].parts['PART-1']
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
    # Create an analysis step
    mdb.models['test'].StaticStep(name='Step-1', previous='Initial', maxNumInc=1000, initialInc=0.01, minInc=0.001, nlgeom=ON)
    mdb.models['test'].fieldOutputRequests['F-Output-1'].setValues(numIntervals=20)
    mdb.models['test'].historyOutputRequests['H-Output-1'].setValues(numIntervals=20)
    
    # Add binding constraints for top and bottom surfaces
    # for node in region1.nodes:
    # a = mdb.models['test'].rootAssembly
    # s1 = a.instances['plate-2'].faces
    # side1Faces1 = s1.getSequenceFromMask(mask=('[#1 ]', ), )
    # region1=a.Surface(side1Faces=side1Faces1, name='m_Surf-1')
    # a = mdb.models['test'].rootAssembly
    # n1 = a.instances['PART-1-1'].nodes
    # nodes1 = n1.getByBoundingBox(xMin=-20.1, xMax=-19.9, yMin=-21, yMax=21, zMin=-21.0, zMax=21.0)
    # region2=a.Set(nodes=nodes1, name='s_Set-1')
    # mdb.models['test'].Tie(name='Constraint-1', main=region1, secondary=region2, 
    #     positionToleranceMethod=COMPUTED, adjust=OFF, tieRotations=ON, 
    #     constraintEnforcement=NODE_TO_SURFACE, thickness=ON)

    # a = mdb.models['test'].rootAssembly
    # s2 = a.instances['plate-1'].faces
    # side2Faces2 = s2.getSequenceFromMask(mask=('[#1 ]', ), )
    # region3=a.Surface(side2Faces=side2Faces2, name='m_Surf-2')
    # a = mdb.models['test'].rootAssembly
    # n2 = a.instances['PART-1-1'].nodes
    # nodes2 = n2.getByBoundingBox(xMin=19.9, xMax=20.1, yMin=-21, yMax=21, zMin=-21.0, zMax=21.0)
    # region4=a.Set(nodes=nodes2, name='s_Set-2')
    # mdb.models['test'].Tie(name='Constraint-2', main=region3, secondary=region4, 
    #     positionToleranceMethod=COMPUTED, adjust=OFF, tieRotations=ON, 
    #     constraintEnforcement=NODE_TO_SURFACE, thickness=ON)

    # Add contact constraints for top and bottom surfaces
    # for node in region1.nodes:
    # mdb.models['test'].ContactProperty('IntProp-1')
    # mdb.models['test'].interactionProperties['IntProp-1'].NormalBehavior(
    #     pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)
    # mdb.models['test'].interactionProperties['IntProp-1'].TangentialBehavior(
    #     formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
    #     pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
    #     0.05, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, fraction=0.005, elasticSlipStiffness=None)
    # a = mdb.models['test'].rootAssembly
    # s1 = a.instances['plate-2'].faces
    # side1Faces1 = s1.getSequenceFromMask(mask=('[#1 ]', ), )
    # region1=a.Surface(side1Faces=side1Faces1, name='m_Surf-1')
    # a = mdb.models['test'].rootAssembly
    # n1 = a.instances['PART-1-1'].nodes
    # nodes1 = n1.getByBoundingBox(xMin=-20.1, xMax=-19.9, yMin=-21, yMax=21, zMin=-21.0, zMax=21.0)
    # region2=a.Set(nodes=nodes1, name='s_Set-1')
    # mdb.models['test'].SurfaceToSurfaceContactStd(name='Int-1', 
    #     createStepName='Step-1', main=region1, secondary=region2, sliding=FINITE, 
    #     thickness=ON, interactionProperty='IntProp-1', adjustMethod=NONE, 
    #     initialClearance=OMIT, datumAxis=None, clearanceRegion=None)
    
    # for node in region2.nodes:
    # a = mdb.models['test'].rootAssembly
    # s2 = a.instances['plate-1'].faces
    # side2Faces2 = s2.getSequenceFromMask(mask=('[#1 ]', ), )
    # region3=a.Surface(side2Faces=side2Faces2, name='m_Surf-2')
    # a = mdb.models['test'].rootAssembly
    # n2 = a.instances['PART-1-1'].nodes
    # nodes2 = n2.getByBoundingBox(xMin=19.9, xMax=20.1, yMin=-21, yMax=21, zMin=-21.0, zMax=21.0)
    # region4=a.Set(nodes=nodes2, name='s_Set-2')
    # mdb.models['test'].SurfaceToSurfaceContactStd(name='Int-2', 
    #     createStepName='Step-1', main=region3, secondary=region4, sliding=FINITE, 
    #     thickness=ON, interactionProperty='IntProp-1', adjustMethod=NONE, 
    #     initialClearance=OMIT, datumAxis=None, clearanceRegion=None)

    # Add displacement constraints for six surfaces

    # a = mdb.models['test'].rootAssembly
    # r1 = a.instances['plate-2'].referencePoints
    # refPoints1=(r1[2], )
    # region5 = a.Set(referencePoints=refPoints1, name='Set-3')
    # mdb.models['test'].EncastreBC(name='BC-1', createStepName='Step-1', region=region5, localCsys=None)

    # a = mdb.models['test'].rootAssembly
    # r1 = a.instances['plate-1'].referencePoints
    # refPoints2=(r1[2], )
    # region6 = a.Set(referencePoints=refPoints2, name='Set-4')
    # mdb.models['test'].DisplacementBC(name='BC-2', createStepName='Step-1', 
    #     region=region6, u1=-10.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, 
    #     ur3=0.0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, 
    #     fieldName='', localCsys=None)

    a = mdb.models['test'].rootAssembly
    n1 = a.instances['PART-1-1'].nodes
    nodes1 = n1.getByBoundingBox(xMin=-20.1, xMax=-19.9, yMin=-21, yMax=21, zMin=-21.0, zMax=21.0)
    region2=a.Set(nodes=nodes1, name='s_Set-1')
    mdb.models['test'].DisplacementBC(name='BC-1', createStepName='Step-1', 
        region=region2, u1=0.0, u2=UNSET, u3=UNSET, ur1=0.0, ur2=0.0, 
        ur3=0.0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, 
        fieldName='', localCsys=None)

    a = mdb.models['test'].rootAssembly
    n2 = a.instances['PART-1-1'].nodes
    nodes2 = n2.getByBoundingBox(xMin=19.9, xMax=20.1, yMin=-21, yMax=21, zMin=-21.0, zMax=21.0)
    region4=a.Set(nodes=nodes2, name='s_Set-2')
    mdb.models['test'].DisplacementBC(name='BC-2', createStepName='Step-1', 
        region=region4, u1=-10.0, u2=UNSET, u3=UNSET, ur1=0.0, ur2=0.0,
        ur3=0.0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, 
        fieldName='', localCsys=None)

    a = mdb.models['test'].rootAssembly
    n3 = a.instances['PART-1-1'].nodes
    nodes3 = n3.getByBoundingBox(xMin=-21.0, xMax=21.0, yMin=-21.0, yMax=21.0, zMin=-20.1, zMax=-19.9)
    region7 = a.Set(nodes=nodes3, name='Set-5')
    mdb.models['test'].DisplacementBC(name='BC-3', createStepName='Step-1', 
        region=region7, u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
        localCsys=None)

    a = mdb.models['test'].rootAssembly
    n4 = a.instances['PART-1-1'].nodes
    nodes4 = n4.getByBoundingBox(xMin=-21.0, xMax=21.0, yMin=-21.0, yMax=21.0, zMin=19.9, zMax=20.1)
    region8 = a.Set(nodes=nodes4, name='Set-6')
    mdb.models['test'].DisplacementBC(name='BC-4', createStepName='Step-1', 
        region=region8, u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
        localCsys=None)

    a = mdb.models['test'].rootAssembly
    n5 = a.instances['PART-1-1'].nodes
    nodes5 = n5.getByBoundingBox(xMin=-21.0, xMax=21.0, yMin=19.9, yMax=20.1, zMin=-21.0, zMax=21.0)
    region9 = a.Set(nodes=nodes5, name='Set-7')
    mdb.models['test'].DisplacementBC(name='BC-5', createStepName='Step-1', 
        region=region9, u1=UNSET, u2=0.0, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
        localCsys=None)

    a = mdb.models['test'].rootAssembly
    n6 = a.instances['PART-1-1'].nodes
    nodes6 = n6.getByBoundingBox(xMin=-21.0, xMax=21.0, yMin=-20.1, yMax=-19.9, zMin=-21.0, zMax=21.0)
    region10 = a.Set(nodes=nodes6, name='Set-8')
    mdb.models['test'].DisplacementBC(name='BC-6', createStepName='Step-1', 
        region=region10, u1=UNSET, u2=0.0, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
        localCsys=None)

    # Dividing the flat grid
    # p = mdb.models['test'].parts['plate']
    # p.seedPart(size=2.0, deviationFactor=0.1, minSizeFactor=0.1)
    # p = mdb.models['test'].parts['plate']
    # p.generateMesh()

    # Create a job and submit
    mdb.Job(name='Job-1', model='test', description='', type=ANALYSIS, atTime=None, 
        waitMinutes=0, waitHours=0, queue=None, memory=90, memoryUnits=PERCENTAGE, 
        getMemoryFromAnalysis=True, explicitPrecision=SINGLE, 
        nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
        contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
        resultsFormat=ODB, numThreadsPerMpiProcess=1, multiprocessingMode=DEFAULT, 
        numCpus=16, numDomains=16, numGPUs=8)
    try:
        mdb.jobs['Job-1'].submit()
        mdb.jobs['Job-1'].waitForCompletion()
        time.sleep(5)
        session.viewports['Viewport: 1'].setValues(displayedObject=None)
        a = mdb.models['test'].rootAssembly
        session.viewports['Viewport: 1'].setValues(displayedObject=a)
        session.viewports['Viewport: 1'].assemblyDisplay.setValues(optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
        o = session.openOdb(name='Job-1.odb')
        session.viewports['Viewport: 1'].setValues(displayedObject=o)
        session.viewports['Viewport: 1'].makeCurrent()
        odb = session.odbs['Job-1.odb']
        xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', NODAL, ((COMPONENT, 'RF1'), )), ), operator=ADD, nodeSets=("S_SET-2", ))
        xyp = session.XYPlot('XYPlot-11')
        chartName = xyp.charts.keys()[0]
        chart = xyp.charts[chartName]
        curveList = session.curveSet(xyData=xyList)
        chart.setValues(curvesToPlot=curveList)
        session.charts[chartName].autoColor(lines=True, symbols=True)
        session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
        odb = session.odbs['Job-1.odb']
        x0 = session.xyDataObjects['ADD_RF:RF1']
        session.xyReportOptions.setValues(numDigits=8)
        session.writeXYReport(fileName='gradient_class1_'+str(i)+'.txt', xyData=(x0, ))
        # session.writeXYReport(fileName='gyroid_network_0.15_0.05_0.15_without.txt', xyData=(x0, ))
        del session.xyDataObjects['ADD_RF:RF1']
        a = mdb.models['test'].rootAssembly
        print('gyroid_class1_'+str(i)+' has been completed')
        del mdb.models['test']
        o.close()
        end_time = time.time()
        print('this program takes time:', (end_time-start_time)/60)
        try:
            os.remove('Job-1.lck')
        except Exception as e:
            pass 
    except Exception as e:
#         print('program-gyroid-network'+str(i)+'has errors')
#         del session.xyDataObjects['ADD_RF:RF1']
        
#         # try:
#         #     os.remove('Job-1.lck')
#         #     # o = session.openOdb(name='Job-1.odb')
#         #     # o.close()
#         # except Exception as e:
#         #     pass    
#         del mdb.models['test']

        # continue
    # finally:
        pass
else:
    print('program_'+str(i)+'does not exists')
pass

time2 = time.time()
print('all programs take time:', (time2-time1)/60/60)