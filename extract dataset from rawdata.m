clc; clear; 
filename1 = 'G:\测试数据\TPMS\仿真结果汇总\parameter_gradient_1_added.xlsx';
data1 = readmatrix(filename1);
emptyArray1 = []; emptyArray2 = []; emptyArray5 = [];
emptyArray3 = []; emptyArray4 = []; 
% figure
% x = linspace(0,10,21);
for i = 1:5000
    filename = ['G:\测试数据\TPMS\仿真结果汇总\class1\gradient_class1_',num2str(i),'.txt'];
    num = exist(filename);
    if num == 2
        data = readmatrix(filename);
        if length(data) == 21
            % strain = data(:,1);
            stress = -data(2:21,2)';
            V = [data1(i,1), data1(i,2), data1(i,3)]; 
            w = data1(i,4); relativeVolume = data1(i,6);
            relativeArea = data1(i,8); poreDiameter = data1(i,10); 
            areaMean = data1(i,12); prop = [relativeVolume,relativeArea,poreDiameter,areaMean];
            V1 = round(V,3); w1 = round(w,3); prop = round(prop,3);
            text1 = [V1,w1,prop,stress]; 
            emptyArray1(end+1,:) = text1;
            V2 = round(V,4); w2 = round(w,4); prop = round(prop,4);
            text2 = [V2,w2,prop,stress]; 
            emptyArray2(end+1,:) = text2;
            V3 = round(V,5); w3 = round(w,5); prop = round(prop,5);
            text3 = [V3,w3,prop,stress]; 
            emptyArray3(end+1,:) = text3;
            V4 = round(V,6); w4 = round(w,6); prop = round(prop,6);
            text4 = [V4,w4,prop,stress]; 
            emptyArray4(end+1,:) = text4;
            V5 = round(V,7); w5 = round(w,7); prop = round(prop,7);
            text5 = [V5,w5,prop,stress]; 
            emptyArray5(end+1,:) = text5;

            % force = [0,stress];
            % plot(x,force); hold on
        end
    end
end
emptyArray = [emptyArray1;emptyArray2;emptyArray3;emptyArray4;emptyArray5];