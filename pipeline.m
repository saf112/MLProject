%% GET FEATURES
% goal: extract cos correlation features from all patients
disp('extracting features')
clear all
cd ../../Research ;

TME_filepath='TMA_Core_Files';
allTME=dir(TME_filepath);

pat_table=readtable('clinical_data_july2.csv');
temp=pat_table;
pat_table.Properties.RowNames=pat_table.spotname;

num_features=1485;
features=zeros(num_features,700);

DNEcount=0;
nocells=0;
nan_count=0;
k=1;

%loop for each patient in table
for i=1:1:height(pat_table)
   name=char(temp(i,:).spot_name);
   fname=fullfile(TME_filepath,strcat(name,'.mat'));
    %if file exists (some patients dont have mat files)
   if exist(fname)
       load(fname);
   else
       %disp(['DNE' name])
       pat_table(cellstr(name),:)=[];
       DNEcount=DNEcount+1;
       continue 
   end
   
   if length(bm_data(:,1))<500
       pat_table(cellstr(name),:)=[];
       nocells=nocells+1;
       continue
   end
   
   bmarkers=cellstr(bm_names);
   cellInds=1:4:length(bmarkers);
   bmarkers=bmarkers(cellInds,:);
   %only use biomarkers at the cell level
   data=bm_data(:,cellInds)';
   %remove DAPI BM --> bm 55
   data(55,:)=[];
   bmarkers(55,:)=[];
   %calc cos corr for all biomarkers
   coscorr=1-pdist(data,'cosine');
   if sum(isnan(coscorr))
       pat_table(cellstr(name),:)=[];
       nan_count=nan_count+1;
       continue
   end
   %add feature vector to all features
   features(:,k)=coscorr;
   k=k+1;
end

features(:,k:end)=[];
features=features';
DNEcount
nan_count
nocells

cd ../MLproject/mat_files ;
save('features.mat','features');
save('pat_table.mat','pat_table');

%% GET FEATURE NAMES 
feature_names=cell(1485,1);
count=1;

for i=1:1:length(bmarkers)
    bm1=char(bmarkers(i,:));
    bm1=bm1(13:end);
    for j=i+1:1:length(bmarkers)
        bm2=char(bmarkers(j,:));
        bm2=bm2(13:end);
        name=strcat(strcat(bm1,' and '),strcat(' ',bm2));
        feature_names{count,1}=name;
        count=count+1;
    end
end

save('feature_names.mat','feature_names')

%% SPLIT FEATURES TO TEST/TRAIN FEATURES
%goal: 20% test data, 80% training data
%features: patients x features

load features.mat
load pat_table.mat

labels=pat_table.stage_sumajc;
test_ind=[];
for i=1:1:3
    s=find(labels==i);
    n=length(s);
    ind=randsample(s,floor(n*0.2));
    test_ind=[test_ind;ind];
end
% get test data & labels
test_data=features(test_ind,:);
test_table=pat_table(test_ind,:);

% get train data & labels
train_data=features;
train_data(test_ind,:)=[];
train_table=pat_table;
train_table(test_ind,:)=[];

survival=pat_table.survtime_days;
test_surv=survival(test_ind);
train_surv=survival;
train_surv(test_ind)=[];

save('train_table.mat','train_table')
save('test_table.mat','test_table')
save('train_surv.mat','train_surv')
save('train_data.mat','train_data')
save('test_surv.mat','test_surv')
save('test_data.mat','test_data')
%% TRAIN DECISION TREE(s) --STAGE PREDICTION
%goal: using train data & labels, train and optimatize decision tree(s)
load train_table.mat
load test_table.mat

train_labels=train_table.stage_sumajc;
test_labels=test_table.stage_sumajc;

%cross val default: 10 fold cross validation
model=fitctree(train_data,train_labels,'CrossVal','on','MinLeafSize',39)

%to view k fold decision trees
% for i=1:1:length(model.Trained)
%     view(model.Trained{i},'Mode','graph')
% end

%training accuracy 
train_acc=zeros(length(model.Trained),1);
for i=1:1:length(model.Trained)
    p=predict(model.Trained{i},train_data);
    acc=length(find(p==train_labels))/length(train_labels);
    train_acc(i,1)=acc;
end
disp('avg train acc')
avg_train_acc=mean(train_acc)

%testing accuracy 
test_acc=zeros(length(model.Trained),1);
for i=1:1:length(model.Trained)
    p=predict(model.Trained{i},test_data);
    acc=length(find(p==test_labels))/length(test_labels);
    test_acc(i,1)=acc;
end
disp('avg train acc')
avg_test_acc=mean(test_acc)

%% FEATURES FOR RECURRENCE TIME PREDICTION & train/test data
load pat_table.mat;
load features.mat;

chemo_table=pat_table(find(pat_table.chemo==0),:);
chemo_features=features(find(pat_table.chemo==0),:);

r=round(chemo_table.recurtime_days/365);
% chemo_table=chemo_table(find(r<=5),:);
% chemo_features=chemo_features(find(r<=5),:);

labels=chemo_table.stage_sumajc;
ctest_ind=[];
for i=1:1:3
    s=find(labels==i);
    n=length(s);
    ind=randsample(s,floor(n*0.2));
    ctest_ind=[ctest_ind;ind];
end
% get test data & labels
ctest_data=chemo_features(ctest_ind,:);
ctest_table=chemo_table(ctest_ind,:);

% get train data & labels
ctrain_data=chemo_features;
ctrain_data(ctest_ind,:)=[];
ctrain_table=chemo_table;
ctrain_table(ctest_ind,:)=[];

save('chemo_table.mat','chemo_table');
save('ctrain_table.mat','ctrain_table');
save('ctest_table.mat','ctest_data');
save('chemo_features.mat','chemo_features');
save('ctest_data.mat','ctest_data');
save('ctrain_data.mat','ctrain_data');

%% TRAIN DECISION TREE -- TIME TO RECURRENCE
load ctrain_table;
load ctest_table;
load ctrain_data;
load ctest_data;

ctrain_table.recurtime_days=round(ctrain_table.recurtime_days./365);
ctest_table.recurtime_days=round(ctest_table.recurtime_days./365);

ctrain_labels=ctrain_table.recurtime_days;
ctest_labels=ctest_table.recurtime_days;

num_trees=10;
recurr_model=TreeBagger(num_trees,ctrain_data,ctrain_labels)

p=predict(recurr_model,ctrain_data);
p1=zeros(length(p),1);
for i=1:1:length(p)
    p1(i,1)=str2num(cell2mat(p(i,1)));
end
p=p1;
acc=length(find(p==ctrain_labels))/length(ctrain_labels);
disp('train acc')
acc

%testing accuracy 
p=predict(recurr_model,ctest_data);
p1=zeros(length(p),1);
for i=1:1:length(p)
    p1(i,1)=str2num(cell2mat(p(i,1)));
end
p=p1;
acc=length(find(p==ctest_labels))/length(ctest_labels);
disp('test acc')
acc

%% GET NEIGHBORHOOD CELL FEATURES FOR A RANDOM PATIENT
%you can load train_table.mat or test_table.mat if you want to make sure to
%get a patient from one of those sets
%pick a patient from chemo table that way you can show maps for recurr time
%too
load chemo_table.mat

cd ../../Research ;
filepath='TMA_Core_Files';

num_pat=height(chemo_table);
cell_thres=500;
num_cells=0;
while num_cells<cell_thres
    pat_num=randsample([1:1:num_pat],1);
    name=char(chemo_table(pat_num,:).spot_name);
    fname=fullfile(filepath,strcat(name,'.mat'));
    load(fname);
    num_cells=length(bm_data(:,1));
    if num_cells<cell_thres
        continue
    end

h=75;
num_features=1485;
cell_features=zeros(num_cells,num_features);
nan_count=0;
delcells=[];
k=1;
for j=1:1:num_cells
    cellx=x(1,j);
    celly=y(1,j);
    distx=abs(x'-cellx);
    disty=abs(y'-celly);
    ind=find(distx<=h & disty<=h);

    bmarkers=cellstr(bm_names);
    cellInds=1:4:length(bmarkers);
    
    cells=bm_data(ind,cellInds);
    %remove dapi
    cells(:,55)=[];
    
    cells_corr=1-pdist(cells','cosine');
    
    if sum(isnan(cells_corr))
        nan_count=nan_count+1;
        delcells=[delcells;j];
        continue
    end
    
    cell_features(k,:)=cells_corr;
    k=k+1;
end
num_cells=length(cell_features(:,1));

end
cell_features(k:end,:)=[];
x(delcells)=[];
y(delcells)=[];
area(delcells)=[];
labels=chemo_table.stage_sumajc;
stage=labels(pat_num);
pat_recurr=chemo_table.recurtime_days(pat_num);
pat_recurr=pat_recurr/365;
nan_count
num_cells

cd ../MLproject/mat_files
save('pat_index.mat','pat_num');
save('patient_name.mat','name');
save('patient_cell_features.mat','cell_features');
save('pat_stage.mat','stage');
save('pat_x.mat','x');
save('pat_y.mat','y');
save('pat_area.mat','area');


%% PREDICT STAGE BASED ON CELL FEATURES AND PLOT ACCORDING TO LOCATION 
load pat_index.mat
load patient_cell_features.mat
load pat_stage.mat
load pat_x.mat
load pat_y.mat
load patient_name.mat
load pat_area.mat
load TSNE_points.mat
load pat_table.mat

labels=pat_table.stage_sumajc;

pred=zeros(length(cell_features(:,1)),length(model.Trained));
for i=1:1:length(model.Trained)
    p=predict(model.Trained{i},cell_features);
    pred(:,i)=p;
end
predictions=round(mean(pred,2));
c=[1 0 0;0 1 0;0 0 1];
x=x';
y=y';
s1=find(predictions==1);
s1x=x(s1);
s1y=y(s1);
s2=find(predictions==2);
s2x=x(s2);
s2y=y(s2);
s3=find(predictions==3);
s3x=x(s3);
s3y=y(s3);
a1=area(s1);
a2=area(s2);
a3=area(s3);

p=predict(recurr_model,cell_features);
p1=zeros(length(p),1);
for i=1:1:length(p)
    p1(i,1)=str2num(cell2mat(p(i,1)));
end
recurr_pred=p1;

figure;
scatter(x,y,0.05*area,recurr_pred,'filled')
name=strrep(name, '_',' ');
t=strcat(name, ' - time to reccur ');
t=strcat(t, num2str(pat_recurr));
title(t);
colorbar;

figure;
scatter(s1x,s1y,0.05*a1,'r','filled')
%plot(s1x,s1y,'r.','MarkerSize',12)
hold on
scatter(s2x,s2y,0.05*a2,'g','filled')
hold on
scatter(s3x,s3y,0.05*a3,'b','filled')
legend('stage 1','stage 2', 'stage 3')
name=strrep(name, '_',' ');
t=strcat(name, ' - stage ');
t=strcat(t, num2str(stage));
title(t);

s1=find(labels==1);
t1=TSNE_points(s1,:);
s2=find(labels==2);
t2=TSNE_points(s2,:);
s3=find(labels==3);
t3=TSNE_points(s3,:);

figure;
plot(t1(:,1),t1(:,2),'r.','MarkerSize',12)
hold on
plot(t2(:,1),t2(:,2),'g.','MarkerSize',12)
hold on
plot(t3(:,1),t3(:,2),'b.','MarkerSize',12)
hold on
plot(TSNE_points(pat_num,1),TSNE_points(pat_num,2),'ko','MarkerSize',13)
legend('stage 1','stage 2', 'stage 3', name)