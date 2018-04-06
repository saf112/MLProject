%% GET FEATURES
% goal: extract cos correlation features from all patients
disp('extracting features')
clear all
cd ../../LAB ;

TME_filepath='TMA_Core_Files';
allTME=dir(TME_filepath);

pat_table=readtable('clinical_data_july2.csv');
temp=pat_table;
pat_table.Properties.RowNames=pat_table.spotname;

num_features=1540;
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
   
   bmarkers=cellstr(bm_names);
   cellInds=1:4:length(bmarkers);
   %only use biomarkers at the cell level
   data=bm_data(:,cellInds)';
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
labels=pat_table.stage_sumajc;
DNEcount
nan_count

cd ../MachineLearning/Project ;
save('features.mat','features');
save('labels.mat','labels');
save('pat_table.mat','pat_table');

%% SPLIT FEATURES TO TEST/TRAIN FEATURES
%goal: 20% test data, 80% training data
%features: patients x features
%labels: patients x 1
load features.mat
load labels.mat

test_ind=[];
for i=1:1:3
    s=find(labels==i);
    n=length(s);
    ind=randsample(s,floor(n*0.2));
    test_ind=[test_ind;ind];
end
% get test data & labels
test_labels=labels(test_ind,1);
test_data=features(test_ind,:);
% get train data & labels
train_data=features;
train_labels=labels;
train_data(test_ind,:)=[];
train_labels(test_ind,:)=[];

%% TRAIN DECISION TREE(s) 
%goal: using train data & labels, train and optimatize decision tree(s)

%cross val default: 10 fold cross validation
model=fitctree(train_data,train_labels,'CrossVal','on')

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

%% GET NEIGHBORHOOD CELL FEATURES FOR A RANDOM PATIENT
load pat_table.mat

cd ../../LAB ;
filepath='TMA_Core_Files';


num_pat=height(pat_table);
pat_num=randi([1 num_pat],1);
name=char(pat_table(pat_num,:).spot_name);
fname=fullfile(TME_filepath,strcat(name,'.mat'));
load(fname);
num_cells=length(bm_data(:,1));

h=100;
num_features=1540;
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
    
    cells_corr=1-pdist(cells','cosine');
    
    if sum(isnan(cells_corr))
        nan_count=nan_count+1;
        delcells=[delcells;j];
        continue
    end
    
    cell_features(k,:)=cells_corr;
    k=k+1;
end
cell_features(k:end,:)=[];
x(delcells)=[];
y(delcells)=[];
labels=pat_table.stage_sumajc;
stage=labels(pat_num);
nan_count

cd ../MachineLearning/Project
save('patient_name.mat','name');
save('cell_features.mat','cell_features');
save('pat_stage.mat','stage');
save('pat_x.mat','x');
save('pat_y.mat','y');

%% PREDICT STAGE BASED ON CELL FEATURES AND PLOT ACCORDING TO LOCATION 

load cell_features.mat
load pat_stage.mat
load pat_x.mat
load pat_y.mat

pred=zeros(length(cell_features(:,1)),length(model.Trained));
for i=1:1:length(model.Trained)
    p=predict(model.Trained{i},cell_features);
    pred(:,i)=p;
end
predictions=round(mean(pred,2));
c=[1 0 0;0 1 0;0 0 1];
stage
scatter(x',y',[],c(predictions,:))