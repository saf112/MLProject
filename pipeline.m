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
DNEcount
nan_count

cd ../../Machine Learning/Project ;
save('features.mat','features')

%% split data to train / CV / testing 
features=features';
temp=[];
cv=[];
test=[];
for i=1:1:3
    s=find(labels==i);
    n=length(s);
    hold=randsample(s,floor(n*0.2));
%     l=length(hold);
%     h1=hold(1:l/2);
%     h2=hold(l/2 +1:end);
%     
%     cv=[cv;h1];
    h2=hold;
    test=[test;h2];
end
% hold=[cv;test];
% cv_labels=labels(cv,1);
% cv=features(cv,:);
hold=test;
test_labels=labels(test,1);
test=features(test,:);
features(hold,:)=[];
labels(hold,:)=[];

%% train random forest
model=fitctree(features,labels)
pred=predict(features)


%% get cell nbhd features for random patient
cd ../../LAB ;
filepath='TMA_Core_Files';
load pat.mat;

num_pat=height(pat_table);
pat_num=randi([1 num_pat],1);
name=char(pat_table(pat_num,:).spot_name);
fname=fullfile(TME_filepath,strcat(name,'.mat'));
load(fname);
num_cells=length(bm_data(:,1));

h=150;
num_features=24976;
cell_features=zeros(num_cells,num_features);
for j=1:1:num_cells
    j
    cellx=x(1,j);
    celly=y(1,j);
    distx=abs(x'-cellx);
    disty=abs(y'-celly);
    ind=find(distx<=h & disty<=h);
    
    %disp('Cells found in nbhd: ')
    %length(ind)
    
    cells=bm_data(ind,:);
    bmarkers=cellstr(bm_names);
    cellInds=1:4:length(bmarkers);
    subInds=1:length(bmarkers);
    subInds(cellInds)=[];
    
    cells_corr=squareform(1-pdist(cells','cosine'));
%     cells_epithelial=epithelial(1,ind);
%     epi=find(cells_epithelial==1);
%     strom=find(cells_epithelial==2);
%     epi_data=cells(epi,subInds);
%     strom_data=cells(strom,cellInds);
%     
%     epi_corr=squareform(1-pdist(epi_data','cosine'));
%     strom_corr=squareform(1-pdist(strom_data','cosine'));
%    
%     [meanMat, stdMat]=meanstd_corrmat(epi_corr,strom_corr, epi_data, strom_data);
%    
    N=length(cells_corr(:,1));
    f_vec=zeros((N*(N-1))/2,1);
    c=1;
    for i=1:1:N
        for m=i+1:1:N
            f_vec(c,1)=cells_corr(i,m);
            c=c+1;
        end
    end

    cell_features(j,:)=f_vec';
end