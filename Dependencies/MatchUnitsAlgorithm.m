%function  [UniqueID, Prob] = MatchUnitsAlgorithm(clusinfo,AllQMsPaths)
%% Match units on neurophysiological evidence
% Input:
% - clusinfo (this is phy output, see also prepareinfo/spikes toolbox)
% - AllQMs: cell struct with QM for every session

% Output:
% - UniqueID (Units with large overlap in QMs are likely the same unit, and
% will share a UniqueID)
% - Prob: Probability of all units to be the same as every other unit

% Matching occurs on:
% - Template correlation
% - # peaks
% - # throughs
% - # Non-somatic
% - waveform amplitude

% Testing the match:
% - Height on probe

%% Parameters
sampleamount = 500; % Nr. waveforms to include
spikeWidth = 83; % in sample space (time)
TakeChannelRadius = 100; %in micron around max channel

%% Extract all clusters
AllClusterIDs = clusinfo.cluster_id;
nses = length(AllQMsPaths);
OriginalClusID = AllClusterIDs;
UniqueID = 1:length(AllClusterIDs); % Initial assumption: All clusters are unique
Good_Idx = find(clusinfo.Good_ID); %Only care about good units at this point
GoodRecSesID = clusinfo.RecSesID(Good_Idx);
% qmetrics
for sesid=1:nses
    qMetric = load(AllQMsPaths{sesid});
    qMetric = qMetric.qMetric;
    Template = qMetric.tempWv;
    pathparts = strsplit(AllRawPaths{sesid},'\');
    rawdatapath = dir(fullfile('\\',pathparts{1:end-1},'templates._jf_Multi_RawWaveforms.npy'))
    if isempty(rawdatapath)
        rawdatapath = dir(fullfile(pathparts{1:end-1},'templates._jf_Multi_RawWaveforms.npy'))
    end
    PeakChan = readNPY(fullfile(rawdatapath.folder,'templates._jf_rawWaveformPeakChannels.npy'));
    if sesid==1
        AllTemplates = nan(length(Good_Idx),size(Template,2),'single');
        PeakLocs = cell(length(Good_Idx),1);
        TroughLocs = cell(length(Good_Idx),1);
        SpatialDecaySlope = nan(length(Good_Idx),1);
        SpatialDecayPoints = nan(length(Good_Idx),1);
        waveformduration = nan(length(Good_Idx),1);
        nPeaks = nan(length(Good_Idx),1);
        nTroughs = nan(length(Good_Idx),1);
        MaxChan = nan(length(Good_Idx),1);      
    end
    AllTemplates(GoodRecSesID==sesid,:) = Template(logical(clusinfo.Good_ID(recsesAll==sesid)),:);
    PeakLocs(GoodRecSesID==sesid) = qMetric.peakLocs(logical(clusinfo.Good_ID(recsesAll==sesid)));
    TroughLocs(GoodRecSesID==sesid) = qMetric.troughLocs(logical(clusinfo.Good_ID(recsesAll==sesid)));
    SpatialDecaySlope(GoodRecSesID==sesid) = qMetric.spatialDecaySlope(logical(clusinfo.Good_ID(recsesAll==sesid)));
    SpatialDecayPoints(GoodRecSesID==sesid) = qMetric.spatialDecayPoints(logical(clusinfo.Good_ID(recsesAll==sesid)));
    waveformduration(GoodRecSesID==sesid) = qMetric.waveformDuration(logical(clusinfo.Good_ID(recsesAll==sesid)));
    nPeaks(GoodRecSesID==sesid) = qMetric.nPeaks(logical(clusinfo.Good_ID(recsesAll==sesid)));
    nTroughs(GoodRecSesID==sesid) = qMetric.nTroughs(logical(clusinfo.Good_ID(recsesAll==sesid)));
    MaxChan(GoodRecSesID==sesid)=PeakChan(logical(clusinfo.Good_ID(recsesAll==sesid)));
end
SessionSwitch = find(GoodRecSesID==2,1,'first');
% Define day stucture
[X,Y]=meshgrid(recsesAll(Good_Idx));
nclus = length(Good_Idx);

%% Waveform template correlations
TemplateMSE = arrayfun(@(X) arrayfun(@(Y) nanmean((AllTemplates(X,:)-AllTemplates(Y,:)).^2),1:nclus,'UniformOutput',0),1:nclus,'UniformOutput',0);
TemplateMSE = cell2mat(cat(1,TemplateMSE{:}));
% Typically a log, normalize first
TemplateMSE = log10(TemplateMSE);
TemplateMSE(isinf(abs(TemplateMSE)))=quantile(TemplateMSE(~isinf(abs(TemplateMSE))),0.01);
TemplateMSE =1-(TemplateMSE-quantile(TemplateMSE(:),0.01))./(quantile(TemplateMSE(:),0.99)-quantile(TemplateMSE(:),0.01));
TemplateMSE(TemplateMSE>1)=1;
TemplateMSE(TemplateMSE<0)=0;

%% Load raw waveforms
halfWidth = floor(spikeWidth / 2);
dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
nChannels = param.nChannels;
% Take geographically close channels (within 50 microns!), not just index!
ChanIdx = arrayfun(@(X) find(cell2mat(arrayfun(@(Y) norm(channelpos(MaxChan(X),:)-channelpos(Y,:)),1:size(channelpos,1),'UniformOutput',0))<TakeChannelRadius)...
    ,1:nclus,'UniformOutput',0); %Averaging over 10 channels helps with drift
nChan = max(cellfun(@length,ChanIdx));
timercounter = tic;
fprintf(1,'Extracting raw waveforms. Progress: %3d%%',0)

Currentlyloaded = 0;
for uid = 1:nclus
    pathparts = strsplit(AllRawPaths{GoodRecSesID(uid)},'\');
    rawdatapath = dir(fullfile('\\',pathparts{1:end-1}));
    if isempty(rawdatapath)
        rawdatapath = dir(fullfile(pathparts{1:end-1}));
    end
    if exist(fullfile(rawdatapath(1).folder,'RawWaveforms',['Unit' num2str(AllClusterIDs(Good_Idx(uid))) '_RawSpikes.mat']))
        continue
    end
    fprintf(1,'\b\b\b\b%3.0f%%',uid/nclus*100)

    if ~(GoodRecSesID(uid) == Currentlyloaded) % Only load new memmap if not already loaded
        % Map the data
        spikeFile = dir(AllDecompPaths{GoodRecSesID(uid)});
        try %hacky way of figuring out if sync channel present or not
            n_samples = spikeFile.bytes / (param.nChannels * dataTypeNBytes);
            ap_data = memmapfile(AllDecompPaths{GoodRecSesID(uid)}, 'Format', {'int16', [param.nChannels, n_samples], 'data'});
        catch
            nChannels = param.nChannels - 1;
            n_samples = spikeFile.bytes / (nChannels * dataTypeNBytes);
            ap_data = memmapfile(AllDecompPaths{GoodRecSesID(uid)}, 'Format', {'int16', [nChannels, n_samples], 'data'});
        end
        memMapData = ap_data.Data.data;
        Currentlyloaded = GoodRecSesID(uid);
    end

    % Spike samples
    idx1=(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(uid)) & sp.RecSes == GoodRecSesID(uid)).*round(sp.sample_rate));  % Spike times in samples;
    % Take geographically close channels (within 50 microns!), not just index!
    ChanIdx = find(cell2mat(arrayfun(@(Y) norm(channelpos(MaxChan(uid),:)-channelpos(Y,:)),1:size(channelpos,1),'UniformOutput',0))<TakeChannelRadius); %Averaging over 10 channels helps with drift
    spikeMap = nan(spikeWidth,nChan,sampleamount);

    %Extract raw waveforms on the fly - % Unit uid
    try
        spikeIndicestmp = sort(datasample(idx1,sampleamount,'replace',false));
    catch ME
        spikeIndicestmp = idx1;
    end
    for iSpike = 1:length(spikeIndicestmp)
        thisSpikeIdx = int32(spikeIndicestmp(iSpike));
        if thisSpikeIdx > halfWidth && (thisSpikeIdx + halfWidth) * dataTypeNBytes < spikeFile.bytes % check that it's not out of bounds

            tmp = smoothdata(double(memMapData(ChanIdx,thisSpikeIdx-halfWidth:thisSpikeIdx+halfWidth)),2,'gaussian',5);
            tmp = (tmp - mean(tmp(:,1:10),2))';
            tmp(:,end+1:nChan) = nan(size(tmp,1),nChan-size(tmp,2));
            % Subtract first 10 samples to level spikes
            spikeMap(:,:,iSpike) = tmp;
        end
    end

    %     RawWaveForm(GoodRecSesID==sesid,:,:) = squeeze(cat(1,tmp{:}));
    %     save spikeMap
  
    if ~exist(fullfile(rawdatapath(1).folder,'RawWaveforms'))
        mkdir(fullfile(rawdatapath(1).folder,'RawWaveforms'))
    end
    save(fullfile(rawdatapath(1).folder,'RawWaveforms',['Unit' num2str(AllClusterIDs(Good_Idx(uid))) '_RawSpikes.mat']),'spikeMap')

end
fprintf('\n')
disp(['Extracting raw waveforms took ' num2str(round(toc(timercounter)./60)) ' minutes for ' num2str(nclus) ' units'])

%% Projected location %Calculate the projected location in 2D
% Re extract channel locations
timercounter = tic;
ChanIdx = arrayfun(@(X) find(cell2mat(arrayfun(@(Y) norm(channelpos(MaxChan(X),:)-channelpos(Y,:)),1:size(channelpos,1),'UniformOutput',0))<TakeChannelRadius)...
    ,1:nclus,'UniformOutput',0); %Averaging over 10 channels helps with drift
ProjectedLocation = nan(2,nclus);
ProjectedWaveform = nan(spikeWidth,nclus);
AddWeights = nan(spikeWidth,nclus);
MaxChannel = nan(1,nclus);
for uid = 1:nclus
    pathparts = strsplit(AllRawPaths{GoodRecSesID(uid)},'\');
    rawdatapath = dir(fullfile('\\',pathparts{1:end-1}));
    if isempty(rawdatapath)
        rawdatapath = dir(fullfile(pathparts{1:end-1}));
    end

    % Load raw data
    SM1=load(fullfile(rawdatapath(1).folder,'RawWaveforms',['Unit' num2str(num2str(AllClusterIDs(Good_Idx(uid)))) '_RawSpikes.mat']));
    SM1 = SM1.spikeMap; %Average across these channels

    % Mean location:
    Locs = channelpos(ChanIdx{uid},:);
    mu = sum(repmat(nanmax(abs(nanmean(SM1(:,1:size(Locs,1),:),3)),[],1),size(Locs,2),1).*Locs',2)./sum(repmat(nanmax(abs(nanmean(SM1(:,1:size(Locs,1),:),3)),[],1),size(Locs,2),1),2);
    ProjectedLocation(:,uid)=mu;

    % Mean waveform - first extract the 'weight' for each channel, based on
    % how close they are to the projected location (closer = better)
    LocationWeight = sqrt(nansum(abs(Locs-ProjectedLocation(:,uid)').^2,2)); 
    % Waveform at closest location
    [~,minidx] = nanmin(LocationWeight);
    WavMax = nanmean(SM1(:,minidx,:),3);
    MaxChannel(uid) = minidx;
    % Waveform at nearest location
    maxidx = minidx+2;
    WavMin = nanmean(SM1(:,maxidx,:),3);

    % Difference between these two, normalized by their distance, so we get
    % the 'waveform decline' per micron
    WavDiff = (WavMax-WavMin)./sqrt(nansum(abs(Locs(minidx,:)-Locs(maxidx,:)).^2));
    %     Now multiplied by the maximum distance, should be the full signal
    AddWeights(:,uid) = LocationWeight(minidx).*WavDiff;
    ProjectedWaveform(:,uid) = WavMax+AddWeights(:,uid); % The projected location should be wavMax+decline*distance from max waveform
end

% LocDist
LocDist = nan(nclus,nclus);
for uid = 1:nclus
    for uid2=1:nclus
        LocDist(uid,uid2) = pdist(ProjectedLocation(:,[uid,uid2])');
    end
end
LocDistNorm = 1-((LocDist-nanmin(LocDist(:)))./(nanmax(LocDist(:))-nanmin(LocDist(:))));
disp(['Extracting projected location took ' num2str(round(toc(timercounter)./60)) ' minutes for ' num2str(nclus) ' units'])


%% odd versus even spikes within versus between
RawWVMSE = nan(nclus,nclus);
WVCorr = nan(nclus,nclus);
timercounter = tic;
fprintf(1,'Computing waveform similarity. Progress: %3d%%',0)
for uid = 1:nclus
    fprintf(1,'\b\b\b\b%3.0f%%',uid/nclus*100)

    pathparts = strsplit(AllRawPaths{GoodRecSesID(uid)},'\');
    rawdatapath = dir(fullfile('\\',pathparts{1:end-1}));
    if isempty(rawdatapath)
        rawdatapath = dir(fullfile(pathparts{1:end-1}));
    end

    SM1=load(fullfile(rawdatapath(1).folder,'RawWaveforms',['Unit' num2str(num2str(AllClusterIDs(Good_Idx(uid)))) '_RawSpikes.mat']));
    SM1 = squeeze(SM1.spikeMap(:,MaxChannel(uid),:)); %Take maximum channel
    parfor uid2 = uid:nclus
        pathparts = strsplit(AllRawPaths{GoodRecSesID(uid2)},'\');
        rawdatapath = dir(fullfile('\\',pathparts{1:end-1}));
        if isempty(rawdatapath)
            rawdatapath = dir(fullfile(pathparts{1:end-1}));
        end

        SM2=load(fullfile(rawdatapath(1).folder,'RawWaveforms',['Unit' num2str(num2str(AllClusterIDs(Good_Idx(uid2)))) '_RawSpikes.mat']));
        SM2 = squeeze(SM2.spikeMap(:,MaxChannel(uid2),:)); %Average across these channels

        % SimilarityRatio
        RawWVMSE(uid,uid2) = nanmean(((nanmean(SM1(:,1:2:end),2)+AddWeights(:,uid))-(nanmean(SM2(:,2:2:end),2)+AddWeights(:,uid2))).^2)./...
            nanmean(nanmean(cat(2,(nanmean(SM1(:,1:2:end),2)+AddWeights(:,uid)),(nanmean(SM2(:,2:2:end),2)+AddWeights(:,uid2))),2).^2);        %Correlation
        WVCorr(uid,uid2) = corr((nanmean(SM1(:,1:2:end),2)+AddWeights(:,uid)),(nanmean(SM2(:,2:2:end),2)+AddWeights(:,uid2)));
    end
end
fprintf('\n')

% Normalize distribution
RawWVMSElog = log10(RawWVMSE);
% Mirror these
for uid2 = 1:nclus
    for uid=uid2+1:nclus
        RawWVMSElog(uid,uid2)=RawWVMSElog(uid2,uid);
        WVCorr(uid,uid2) = WVCorr(uid2,uid);
    end
end

% Normalize each row
% WavformSimilarity = arrayfun(@(X) (RawWVMSElog(X,:)-min(RawWVMSElog(X,:)))./(max(RawWVMSElog(X,:))-min(RawWVMSElog(X,:))),1:size(RawWVMSElog,2),'Uni',0);
% WavformSimilarity = cat(1,WavformSimilarity{:});
WavformSimilarity = (RawWVMSElog-quantile(RawWVMSElog(:),0.01))./(quantile(RawWVMSElog(:),0.99)-quantile(RawWVMSElog(:),0.01));
WavformSimilarity(WavformSimilarity<0)=0;
WavformSimilarity(WavformSimilarity>1)=1;
WavformSimilarity = 1-WavformSimilarity;

% Make WVCorr a normal distribution
WVCorr = erfinv(WVCorr);
WVCorr = (WVCorr-nanmin(WVCorr(:)))./(nanmax(WVCorr(:))-nanmin(WVCorr(:)));
disp(['Calculating waveform similarity took ' num2str(round(toc(timercounter)./60)) ' minutes for ' num2str(nclus) ' units'])


%% QM Scores
% PeakLocs = cell(length(Good_Idx),1);
% TroughLocs = cell(length(Good_Idx),1);
% SpatialDecaySlope = nan(length(Good_Idx),1);
% % SpatialDecayPoints = nan(length(Good_Idx),1);
% % waveformduration = nan(length(Good_Idx),1);
% nPeaks = nan(length(Good_Idx),1);
% nTroughs = nan(length(Good_Idx),1);
% Define threshold
timercounter = tic;
waveformdurationcomp = arrayfun(@(Y) arrayfun(@(X) (abs(waveformduration(Y)-waveformduration(X))),1:nclus,'UniformOutput',0),1:nclus,'UniformOutput',0);
waveformdurationcomp = cell2mat(cat(1,waveformdurationcomp{:}));
waveformdurationcomp = 1-(waveformdurationcomp./nanmax(waveformdurationcomp(:))); % Divide by maximum value and turn around 1 and 0


MaxChanloccomp = arrayfun(@(Y) arrayfun(@(X) (norm(MaxChan(Y)-MaxChan(X))),1:nclus,'UniformOutput',0),1:nclus,'UniformOutput',0);
MaxChanloccomp = cell2mat(cat(1,MaxChanloccomp{:}));
MaxChanloccomp = 1-(MaxChanloccomp./nanmax(MaxChanloccomp(:)));% Divide by maximum value and turn around 1 and 0

peaklocscomp = arrayfun(@(Y) arrayfun(@(X)  (abs(nanmean(PeakLocs{X})-nanmean(PeakLocs{Y}))),1:nclus,'UniformOutput',0),1:nclus,'UniformOutput',0);
peaklocscomp = cell2mat(cat(1,peaklocscomp{:}));
peaklocscomp = 1-(peaklocscomp./nanmax(peaklocscomp(:)));% Divide by maximum value and turn around 1 and 0

spatialdecaypointscomp = arrayfun(@(Y) arrayfun(@(X) (abs(SpatialDecayPoints(Y)-SpatialDecayPoints(X))),1:nclus,'UniformOutput',0),1:nclus,'UniformOutput',0);
spatialdecaypointscomp = cell2mat(cat(1,spatialdecaypointscomp{:}));
% Transform
spatialdecaypointscomp = sqrt(spatialdecaypointscomp);
spatialdecaypointscomp = 1-((spatialdecaypointscomp-nanmin(spatialdecaypointscomp(:)))./(nanmax(spatialdecaypointscomp(:))-nanmin(spatialdecaypointscomp(:))));% Divide by maximum value and turn around 1 and 0

spatialdecaysloopcomp = arrayfun(@(Y) arrayfun(@(X) (abs(SpatialDecaySlope(Y)-SpatialDecaySlope(X))),1:nclus,'UniformOutput',0),1:nclus,'UniformOutput',0);
spatialdecaysloopcomp = sqrt(cell2mat(cat(1,spatialdecaysloopcomp{:})));
spatialdecaysloopcomp = 1-((spatialdecaysloopcomp-nanmin(spatialdecaysloopcomp(:)))./(nanmax(spatialdecaysloopcomp(:))-nanmin(spatialdecaysloopcomp(:))));% Divide by maximum value and turn around 1 and 0

% Mirror these
for uid2 = 1:nclus
    for uid=uid2+1:nclus
        waveformdurationcomp(uid,uid2)=waveformdurationcomp(uid2,uid);
        MaxChanloccomp(uid,uid2)=MaxChanloccomp(uid2,uid);
        peaklocscomp(uid,uid2)=peaklocscomp(uid2,uid);
        spatialdecaypointscomp(uid,uid2)=spatialdecaypointscomp(uid2,uid);
        spatialdecaysloopcomp(uid,uid2)=spatialdecaysloopcomp(uid2,uid);
    end
end
disp(['Extracting QM Scores took ' num2str(round(toc(timercounter)./60)) ' minutes for ' num2str(nclus) ' units'])


%% Part 1 --> Sum Total scores components - Have a high threshold to have the 'real matches'
figure('name','Waveform based metrics');
% subplot(2,3,1)
% h=imagesc(MaxChanloccomp,[0 1]);
% title('Channel location distance')
% xlabel('Unit Y')
% ylabel('Unit Z')
% hold on
% line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
% line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
% colormap gray
% colorbar
% makepretty

subplot(2,2,1)
h=imagesc(triu(waveformdurationcomp,1),[0 1]);
title('Waveform duration')
xlabel('Unit Y')
ylabel('Unit Z')
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
colormap(flipud(gray))
colorbar
makepretty

subplot(2,2,2)
h=imagesc(triu(spatialdecaysloopcomp,1),[0 1]);
title('Spatial decay')
xlabel('Unit Y')
ylabel('Unit Z')
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
colormap(flipud(gray))
colorbar
makepretty

subplot(2,2,3)
h=imagesc(triu(WavformSimilarity,1),[0 1]);
title('SimilarityRatioNorm')
xlabel('Unit Y')
ylabel('Unit Z')
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
colormap(flipud(gray))
colorbar
makepretty

% subplot(2,3,5)
% h=imagesc(TemplateMSE,[0 1]);
% title('Template waveform error')
% xlabel('Unit Y')
% ylabel('Unit Z')
% hold on
% line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
% line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
% colormap gray
% colorbar
% makepretty

subplot(2,2,4)
h=imagesc(triu(LocDistNorm,1),[0 1]);
title('Projected location Distance Score')
xlabel('Unit Y')
ylabel('Unit Z')
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
colormap(flipud(gray))
colorbar
makepretty


%% Calculate total score
Scores2Include = {'WavformSimilarity','LocDistNorm','spatialdecaysloopcomp','waveformdurationcomp','peaklocscomp'}

% Scores2Include = {'WavformSimilarity','WVCorr','LocDistNorm','spatialdecaysloopcomp','waveformdurationcomp'}
% Scores2Include = {'WavformSimilarity','LocDistNorm','waveformdurationcomp','MaxChanloccomp','peaklocscomp','spatialdecaypointscomp','spatialdecaysloopcomp','TemplateMSE'}
figure;
if length(Scores2Include)>1
    for scid=1:length(Scores2Include)


        ScoresTmp = Scores2Include;
        ScoresTmp(scid)=[];

        TotalScore = zeros(nclus,nclus);
        for scid2=1:length(ScoresTmp)
            eval(['TotalScore=TotalScore+' ScoresTmp{scid2} ';'])
        end
        base = length(ScoresTmp)-1;

        TotalScoreAcrossDays = TotalScore;
        TotalScoreAcrossDays(X==Y)=nan;


        subplot(2,length(Scores2Include)+1,scid)
        h=imagesc(triu(TotalScore,1),[0 base+1]);
        title(['without ' Scores2Include{scid}])
        xlabel('Unit Y')
        ylabel('Unit Z')
        hold on
        line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
        line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
        colormap(flipud(gray))
        colorbar
        makepretty

        % Thresholds
        ThrsOpt = (length(Scores2Include)-1)*0.9;
        subplot(2,length(Scores2Include)+1,scid+(length(Scores2Include)+1))

        imagesc(triu(TotalScore>ThrsOpt,1))
        hold on
        title(['Thresholding at ' num2str(ThrsOpt)])
        xlabel('Unit Y')
        ylabel('Unit Z')
        hold on
        line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
        line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
        colormap(flipud(gray))
        colorbar
        %     axis square
        makepretty
    end
end
TotalScore = zeros(nclus,nclus);
Predictors = zeros(nclus,nclus,0);
for scid2=1:length(Scores2Include)
    eval(['TotalScore=TotalScore+' Scores2Include{scid2} ';'])
    Predictors = cat(3,Predictors,eval(Scores2Include{scid2}));
end
subplot(2,length(Scores2Include)+1,length(Scores2Include)+1)
h=imagesc(triu(TotalScore,1),[0 length(Scores2Include)]);
title(['Total Score'])
xlabel('Unit Y')
ylabel('Unit Z')
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
colormap(flipud(gray))
colorbar
makepretty

% Make initial threshold --> to be optimized
ThrsOpt = (length(Scores2Include))*0.9;
subplot(2,length(Scores2Include)+1,2*(length(Scores2Include)+1))
imagesc(triu(TotalScore>ThrsOpt,1))
hold on
title(['Thresholding at ' num2str(ThrsOpt)])
xlabel('Unit Y')
ylabel('Unit Z')
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
colormap(flipud(gray))
colorbar
% axis square
makepretty
% Find all pairs
% first factor authentication: score above threshold
ThrsScore = ThrsOpt;
% Take into account:
% takematrix = zeros(size(TotalScore,1),size(TotalScore,2),length(Scores2Include));
% for pid=1:length(Scores2Include)
%     eval(['takematrix(:,:,pid) =' Scores2Include{pid} '>0.90;'])
% end
% 
% [uid,uid2] = find(sum(takematrix,3)==length(Scores2Include)); %Score better than 90% on all values
[uid,uid2] = find(TotalScore>ThrsOpt);
Pairs = cat(2,uid,uid2);
Pairs = sort(Pairs,2);
Pairs=unique(Pairs,'rows');
Pairs(Pairs(:,1)==Pairs(:,2),:)=[]
[PairScore,sortid] = sort(cell2mat(arrayfun(@(X) TotalScore(Pairs(X,1),Pairs(X,2)),1:size(Pairs,1),'Uni',0)),'descend');
Pairs = Pairs(sortid,:);
%For chronic
PairsPyKS = [];
% for uid = 1:nclus
%     pairstmp = find(AllClusterIDs(Good_Idx)==AllClusterIDs(Good_Idx(uid)))';
%     if length(pairstmp)>1
%         PairsPyKS = cat(1,PairsPyKS,pairstmp);
%     end
% end
% PairsPyKS=unique(PairsPyKS,'rows');
% 
% [Int,A,B] = intersect(Pairs,PairsPyKS,'rows');
% PercDetected = size(Int,1)./size(PairsPyKS,1).*100;
% disp(['Detected ' num2str(PercDetected) '% of PyKS matched units'])
% 
% PercOver = (size(Pairs,1)-size(Int,1))./size(PairsPyKS,1)*100;
% disp(['Detected ' num2str(PercOver) '% more units than just PyKS matched units'])

% % interesting: Not detected
% NotB = 1:size(PairsPyKS,1);
% NotB(B) = [];
% OnlyDetectedByPyKS = PairsPyKS(NotB,:);
% 
% % Too much detected
% NotA = 1:size(Pairs,1);
% NotA(A) = [];
% NotdetectedByPyKS = Pairs(NotA,:);

%% Functional score for optimization: compute Fingerprint for the matched units - based on Célian Bimbard's noise-correlation finger print method but applied to across session correlations
% Not every recording day will have the same units. Therefore we will
% correlate each unit's activity with average activity across different
% depths
binsz = 1;
timevec = floor(min(sp.st)):binsz:ceil(max(sp.st));
edges = floor(min(sp.st))-binsz/2:binsz:ceil(max(sp.st))+binsz/2;
disp('Calculate activity correlations')

% Use a bunch of units with high total scores as reference population
Pairs = Pairs(abs(Pairs(:,1)-Pairs(:,2))>=min([nclus-SessionSwitch,]),:);
% Only use every 'unit' once
[val,id1,id2]=unique(Pairs(:,1));
Pairs = Pairs(id1,:);
[val,id1,id2]=unique(Pairs(:,2));
Pairs = Pairs(id1,:);

% Correlation on first day
srMatches = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),Pairs(:,1),'UniformOutput',0);
srMatches = cat(1,srMatches{:});
srAll = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),1:SessionSwitch-1,'UniformOutput',0);
srAll = cat(1,srAll{:});
SessionCorrelation_Pair1 = corr(srMatches(:,1:size(srMatches,2)./2)',srAll(:,1:size(srMatches,2)./2)');
for pid = 1:size(Pairs,1)
    SessionCorrelation_Pair1(pid,Pairs(pid,1)) = nan;
end


figure('name','Cross-correlation Fingerprints')
subplot(1,3,1)
imagesc(SessionCorrelation_Pair1')
colormap(flipud(gray))
xlabel('Candidate Units to be matched')
ylabel('All units')
title('Day 1')
makepretty

% Correlation on first day
srMatches = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),Pairs(:,2),'UniformOutput',0);
srMatches = cat(1,srMatches{:});
srAll = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),SessionSwitch:nclus,'UniformOutput',0);
srAll = cat(1,srAll{:});
SessionCorrelation_Pair2 = corr(srMatches(:,size(srMatches,2)./2+1:end)',srAll(:,size(srMatches,2)./2+1:end)');
for pid = 1:size(Pairs,1)
    SessionCorrelation_Pair2(pid,Pairs(pid,2)-SessionSwitch+1) = nan;
end

subplot(1,3,2)
imagesc(SessionCorrelation_Pair2')
colormap(flipud(gray))
xlabel('Candidate Units to be matched')
ylabel('All units')
title('Day 2')
makepretty

% Add both together
SessionCorrelations = cat(2,SessionCorrelation_Pair1,SessionCorrelation_Pair2)';

% Correlate 'fingerprints'
FingerprintR = arrayfun(@(X) cell2mat(arrayfun(@(Y) corr(SessionCorrelations(X,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))',SessionCorrelations(Y,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))'),1:nclus,'UniformOutput',0)),1:nclus,'UniformOutput',0);
% FingerprintR = arrayfun(@(X) cell2mat(arrayfun(@(Y) corr(SessionCorrelations(X,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))',SessionCorrelation_Pair2(Y,~isnan(SessionCorrelation_Pair1(X,:))&~isnan(SessionCorrelation_Pair2(Y,:)))'),1:size(Pairs,1),'UniformOutput',0)),1:size(Pairs,1),'UniformOutput',0);
FingerprintR = cat(1,FingerprintR{:});

% Remove average within and across day correlations
% tmp1 = (FingerprintR(1:SessionSwitch-1,1:SessionSwitch-1));
% FingerprintR(1:SessionSwitch-1,1:SessionSwitch-1) = FingerprintR(1:SessionSwitch-1,1:SessionSwitch-1)-nanmean(tmp1(:));
% 
% tmp2 = (FingerprintR(SessionSwitch:end,SessionSwitch:end));
% FingerprintR(SessionSwitch:end,SessionSwitch:end) = FingerprintR(SessionSwitch:end,SessionSwitch:end)-nanmean(tmp2(:));
% 
% tmp2 = (FingerprintR(SessionSwitch:end,1:SessionSwitch));
% FingerprintR(SessionSwitch:end,1:SessionSwitch) = FingerprintR(SessionSwitch:end,1:SessionSwitch)-nanmean(tmp2(:));
% 
% tmp2 = (FingerprintR(1:SessionSwitch,SessionSwitch:end));
% FingerprintR(1:SessionSwitch,SessionSwitch:end) = FingerprintR(1:SessionSwitch,SessionSwitch:end)-nanmean(tmp2(:));



subplot(1,3,3)
imagesc(FingerprintR)
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
clim([0.5 1])
colormap(flipud(gray))
xlabel('All units across both days')
ylabel('All units across both days')
title('Correlation Fingerprint')
makepretty

%%
SigMask = zeros(nclus,nclus);
RankScoreAll = nan(size(SigMask));
for pid=1:nclus
    for pid2 = 1:nclus
        if pid2<SessionSwitch
            tmp1 = FingerprintR(pid,1:SessionSwitch-1);
            addthis=0;
        else
            tmp1 = FingerprintR(pid,SessionSwitch:end);
            addthis = SessionSwitch-1;
        end
        [val,ranktmp] = sort(tmp1,'descend');

        tmp1(pid2-addthis)=[];
        
        if FingerprintR(pid,pid2)>nanmean(tmp1)+2*nanstd(tmp1)
            SigMask(pid,pid2)=1;

        end
        RankScoreAll(pid,pid2) = find(ranktmp==pid2-addthis);

        if 0%(any(ismember(Pairs(:,1),pid)) & any(ismember(Pairs(:,2),pid2))) || RankScoreAll(pid,pid2)==1
        tmpfig = figure;
        subplot(2,1,1); plot(SessionCorrelations(pid,:)); hold on; plot(SessionCorrelations(pid2,:))
        xlabel('Unit')
        ylabel('Cross-correlation')
        legend('Day1','Day2')
        makepretty
        title(['r=' num2str(FingerprintR(pid,pid2)) ', rank = ' num2str(RankScoreAll(pid,pid2))])
       

        subplot(2,1,2)
        tmp1 = FingerprintR(1:end,pid);
        tmp1(pid)=[];
        tmp2 = FingerprintR(pid,1:end)';
        tmp2(pid)=[];

        histogram([tmp1;tmp2]);
        hold on
        line([FingerprintR(pid,pid2) FingerprintR(pid,pid2)],get(gca,'ylim'),'color',[1 0 0])
        makepretty

        pause(1)
        delete(tmpfig)
        end

    end
end

% matches = logical(eye(size(Pairs,1)));
% nanidx = isnan(FingerprintR);
% FingerprintR(isnan(FingerprintR))=nanmin(FingerprintR(:)); %Make minimum to not have nans
% [val,rank] = arrayfun(@(X) sort(nanmean([FingerprintR(X,:);FingerprintR(:,X)'],1),'descend'),1:size(Pairs,1),'UniformOutput',0);
% rank = cell2mat(arrayfun(@(X) find(rank{X}==X),1:length(rank),'UniformOutput',0));
% rankscore = zeros(size(Pairs,1),size(Pairs,1));
% rankscore(:,:)=size(Pairs,1)+1;
% rankscore(matches)=rank;
% FingerprintR(nanidx)=nan;
% figure;
% edges = min(FingerprintR(:)):0.1:max(FingerprintR(:));
% histogram(FingerprintR(~matches),edges); hold on; histogram(FingerprintR(matches),edges)
% 
% TotalScoreMatches = arrayfun(@(Y) arrayfun(@(X) TotalScore(X,Y),Pairs(:,1),'UniformOutput',0),Pairs(:,2),'UniformOutput',0);
% TotalScoreMatches = cell2mat(cat(2,TotalScoreMatches{:}));
% % 
% CrossValFP=FingerprintR;
% CrossValFP(matches)=nan;
% ConfirmedMatch = cell2mat(arrayfun(@(X) FingerprintR(X,X)>nanmax(CrossValFP(X,:)),1:size(Pairs,1),'UniformOutput',0));
% matches = double(matches);
% matches(logical(eye(size(matches))))=ConfirmedMatch+1;

figure;

scatter(TotalScore(:),FingerprintR(:),14,RankScoreAll(:),'filled')
colormap(cat(1,[0 0 0],winter))
xlabel('TotalScore')
ylabel('Cross-correlation fingerprint')
makepretty
% h=colorbar;
% h.Label.String = 'Rank';

% matches = double(matches);
% matches(rankscore==1)=2;
% matches = matches/2;
% matches(rankscore>2)=0;
% rankscore(rankscore<3)=1;
% rankscore(rankscore~=1)=0;
% 
% subplot(1,2,2)
% scatter(TotalScoreMatches(:),FingerprintR(:),14,RankScoreAll(:),'filled')
% colormap([0 0 0;summer])
% xlabel('TotalScore')
% ylabel('Cross-correlation fingerprint')
% 
% % legend('Non-matches','matches, no cross-correlation','matches')
% makepretty

% RankScoreAll = 1-((RankScoreAll-nanmin(RankScoreAll(:)))./(nanmax(RankScoreAll(:))-nanmin(RankScoreAll(:))));

%% three ways to define candidate scores
% Total score larger than threshold
CandidatePairs = TotalScore>ThrsOpt & RankScoreAll==1;
CandidatePairs(tril(true(size(CandidatePairs))))=0;

% if exist('PairsPyKS') & ~isempty(PairsPyKS) %If pyks was run on stitched data, take pyks as 'truth'
%     disp('Candidate pairs defined by PyKS stitched')
%     CandidatePairs = false(size(CandidatePairs));
%     for pid = 1:size(PairsPyKS,1)
%         CandidatePairs(PairsPyKS(pid,1),PairsPyKS(pid,2)) = 1;
%     end
% end
% 
% % or use the rank==1 as a requisite. Warning: less data points as the
% % amount of matching units is defined by total score as well
% CandidatePairs = RankScoreAll==1;


%% Prepare naive bayes - inspect probability distributions
figure('name','Parameter Scores');
Edges = [0:0.01:1];
for scid=1:length(Scores2Include)
    eval(['ScoresTmp = ' Scores2Include{scid} ';'])
    ScoresTmp(tril(true(size(ScoresTmp))))=nan;
    subplot(length(Scores2Include),2,(scid-1)*2+1)
    histogram(ScoresTmp(~CandidatePairs),Edges)
    if scid==1
        title('Candidate non-Matches')
    end
    ylabel(Scores2Include{scid})
    makepretty

    subplot(length(Scores2Include),2,scid*2)
    histogram(ScoresTmp(CandidatePairs),Edges)
    if scid==1
        title('Candidate Matches')
    end
    makepretty
end


% 
figure('name','Projected Location Distance to [0 0]')
Dist2Tip = sqrt(nansum(ProjectedLocation.^2,1));
% Dist2TipMatrix = nan(size(CandidatePairs));

Dist2TipMatrix = arrayfun(@(Y) cell2mat(arrayfun(@(X) cat(1,Dist2Tip(X),Dist2Tip(Y)),1:nclus,'Uni',0)),1:nclus,'Uni',0);
Dist2TipMatrix = cat(3,Dist2TipMatrix{:});
Dist2TipMatrix = reshape(Dist2TipMatrix,2,[]);
subplot(1,2,1)
hist3(Dist2TipMatrix(:,~CandidatePairs(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Non-matches')

subplot(1,2,2)
hist3(Dist2TipMatrix(:,CandidatePairs(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Matches')

% Waveform duration
figure('name','WaveDur')
waveformdurationMat = arrayfun(@(Y) cell2mat(arrayfun(@(X) cat(1,waveformduration(X),waveformduration(Y)),1:nclus,'UniformOutput',0)),1:nclus,'UniformOutput',0);
waveformdurationMat = cat(3,waveformdurationMat{:});
subplot(1,2,1)
hist3(waveformdurationMat(:,~CandidatePairs(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Non-matches')

subplot(1,2,2)
hist3(waveformdurationMat(:,CandidatePairs(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Matches')

% SpatialDecaySlope
figure('name','Spatial Decay Slope')
SpatDecMat = arrayfun(@(Y) cell2mat(arrayfun(@(X) cat(1,SpatialDecaySlope(X),SpatialDecaySlope(Y)),1:nclus,'UniformOutput',0)),1:nclus,'UniformOutput',0);
SpatDecMat = cat(3,SpatDecMat{:});
subplot(1,2,1)
hist3(SpatDecMat(:,~CandidatePairs(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Non-matches')

subplot(1,2,2)
hist3(SpatDecMat(:,CandidatePairs(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Matches')


% PeaksLocsComp
% figure('name','PeaksLoc')
% SpatDecMat = arrayfun(@(Y) cell2mat(arrayfun(@(X) cat(1,PeakLocs(X),SpatialDecaySlope(Y)),1:nclus,'UniformOutput',0)),1:nclus,'UniformOutput',0);
% SpatDecMat = cat(3,SpatDecMat{:});
% subplot(1,2,1)
% hist3(SpatDecMat(:,~CandidatePairs(:))')
% xlabel('Unit 1')
% ylabel('Unit 2')
% zlabel('Counts')
% title('Candidate Non-matches')
% 
% subplot(1,2,2)
% hist3(SpatDecMat(:,CandidatePairs(:))')
% xlabel('Unit 1')
% ylabel('Unit 2')
% zlabel('Counts')
% title('Candidate Matches')
%% Naive bays classifier
% Usually this means there's no variance in the match distribution
% (which in a way is great). Create some small variance
flag = 0;
npairs = 0;
MinLoss=1;
npairslatest = 0;
maxrun = 25;
runid=0;
while flag<2 && runid<=maxrun
    flag = 0;
    runid=runid+1
    Tbl = array2table(reshape(Predictors,[],size(Predictors(Pairs(:,1),Pairs(:,2),:),3))); %All parameters
    % Use Rank as 'correct' label
    try
        Mdl = fitcnb(Tbl,CandidatePairs(:));
    catch ME
        disp(ME)
        keyboard
        for id = 1:size(Predictors,3)
            tmp = Predictors(:,:,id);
            if nanvar(tmp(CandidatePairs(:)==1)) == 0
                %Add some noise
                tmp(CandidatePairs(:)==1) = tmp(CandidatePairs(:)==1)+(rand(sum(CandidatePairs(:)==1),1)-0.5)./2;
                tmp(tmp>1)=1;
                Predictors(:,:,id)=tmp;
            end
        end
    end

    % Cross validate on model that uses only prior
    DefaultPriorMdl = Mdl;
    FreqDist = cell2table(tabulate(CandidatePairs(:)==1));
    DefaultPriorMdl.Prior = FreqDist{:,3};
    rng(1);%
    defaultCVMdl = crossval(DefaultPriorMdl);
    defaultLoss = kfoldLoss(defaultCVMdl);

    CVMdl = crossval(Mdl);
    Loss = kfoldLoss(CVMdl);

    if Loss>defaultLoss
        warning('Model doesn''t perform better than chance')
    end
    if round(Loss*10000) >= round(MinLoss*10000)
        flag = flag+1;        
    elseif Loss<MinLoss & Loss>defaultLoss
        MinLoss=Loss;
        BestMdl = Mdl;
    end
    disp(['Loss = ' num2str(round(Loss*10000)/10000)])
    %% Apply naive bays classifier
    Tbl = array2table(reshape(Predictors,[],size(Predictors,3))); %All parameters
    [label, posterior, cost] = predict(Mdl,Tbl);
    label = reshape(label,size(Predictors,1),size(Predictors,2));
    [r, c] = find(label==1 & RankScoreAll==1); %Find matches
    Pairs = cat(2,r,c);
    Pairs = sort(Pairs,2);
    Pairs=unique(Pairs,'rows');
    Pairs(Pairs(:,1)==Pairs(:,2),:)=[];
    MatchProbability = reshape(posterior(:,2),size(Predictors,1),size(Predictors,2));
    figure; imagesc(label)

    %% Functional score for optimization: compute Fingerprint for the matched units - based on Célian Bimbard's noise-correlation finger print method but applied to across session correlations
    % Not every recording day will have the same units. Therefore we will
    % correlate each unit's activity with average activity across different
    % depths
    disp('Recalculate activity correlations')

    % Use a bunch of units with high total scores as reference population
    Pairs = Pairs(abs(Pairs(:,1)-Pairs(:,2))>=min([nclus-SessionSwitch,]),:);
    % Only use every 'unit' once
    [val,id1,id2]=unique(Pairs(:,1));
    Pairs = Pairs(id1,:);
    [val,id1,id2]=unique(Pairs(:,2));
    Pairs = Pairs(id1,:);   
    npairs = size(Pairs,1);
    if npairs==npairslatest
        flag = flag+1;
    end
    npairslatest=npairs;

    disp(['Npairs = ' num2str(npairs)])

    if flag==2
        disp('This will not get any better... quite while ahead')
        break
    end


    % Correlation on first day
    srMatches = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),Pairs(:,1),'UniformOutput',0);
    srMatches = cat(1,srMatches{:});
    srAll = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),1:SessionSwitch-1,'UniformOutput',0);
    srAll = cat(1,srAll{:});
    SessionCorrelation_Pair1 = corr(srMatches(:,1:size(srMatches,2)./2)',srAll(:,1:size(srMatches,2)./2)');
    for pid = 1:size(Pairs,1)
        SessionCorrelation_Pair1(pid,Pairs(pid,1)) = nan;
    end


    figure('name','Cross-correlation Fingerprints')
    subplot(1,3,1)
    imagesc(SessionCorrelation_Pair1')
    colormap(flipud(gray))
    xlabel('Candidate Units to be matched')
    ylabel('All units')
    title('Day 1')
    makepretty

    % Correlation on first day
    srMatches = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),Pairs(:,2),'UniformOutput',0);
    srMatches = cat(1,srMatches{:});
    srAll = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),SessionSwitch:nclus,'UniformOutput',0);
    srAll = cat(1,srAll{:});
    SessionCorrelation_Pair2 = corr(srMatches(:,size(srMatches,2)./2+1:end)',srAll(:,size(srMatches,2)./2+1:end)');
    for pid = 1:size(Pairs,1)
        SessionCorrelation_Pair2(pid,Pairs(pid,2)-SessionSwitch+1) = nan;
    end

    subplot(1,3,2)
    imagesc(SessionCorrelation_Pair2')
    colormap(flipud(gray))
    xlabel('Candidate Units to be matched')
    ylabel('All units')
    title('Day 2')
    makepretty

    % Add both together
    SessionCorrelations = cat(2,SessionCorrelation_Pair1,SessionCorrelation_Pair2)';

    % Correlate 'fingerprints'
    FingerprintR = arrayfun(@(X) cell2mat(arrayfun(@(Y) corr(SessionCorrelations(X,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))',SessionCorrelations(Y,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))'),1:nclus,'UniformOutput',0)),1:nclus,'UniformOutput',0);
    % FingerprintR = arrayfun(@(X) cell2mat(arrayfun(@(Y) corr(SessionCorrelations(X,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))',SessionCorrelation_Pair2(Y,~isnan(SessionCorrelation_Pair1(X,:))&~isnan(SessionCorrelation_Pair2(Y,:)))'),1:size(Pairs,1),'UniformOutput',0)),1:size(Pairs,1),'UniformOutput',0);
    FingerprintR = cat(1,FingerprintR{:});

    subplot(1,3,3)
    imagesc(FingerprintR)
    hold on
    line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
    line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
    clim([0.5 1])
    colormap(flipud(gray))
    xlabel('All units across both days')
    ylabel('All units across both days')
    title('Correlation Fingerprint')
    makepretty
    drawnow

    %
    SigMask = zeros(nclus,nclus);
    RankScoreAll = nan(size(SigMask));
    for pid=1:nclus
        for pid2 = 1:nclus
            if pid2<SessionSwitch
                tmp1 = FingerprintR(pid,1:SessionSwitch-1);
                addthis=0;
            else
                tmp1 = FingerprintR(pid,SessionSwitch:end);
                addthis = SessionSwitch-1;
            end
            [val,ranktmp] = sort(tmp1,'descend');

            tmp1(pid2-addthis)=[];

            if FingerprintR(pid,pid2)>nanmean(tmp1)+2*nanstd(tmp1)
                SigMask(pid,pid2)=1;

            end
            RankScoreAll(pid,pid2) = find(ranktmp==pid2-addthis);

            if 0%(any(ismember(Pairs(:,1),pid)) & any(ismember(Pairs(:,2),pid2))) || RankScoreAll(pid,pid2)==1
                tmpfig = figure;
                subplot(2,1,1); plot(SessionCorrelations(pid,:)); hold on; plot(SessionCorrelations(pid2,:))
                xlabel('Unit')
                ylabel('Cross-correlation')
                legend('Day1','Day2')
                makepretty
                title(['r=' num2str(FingerprintR(pid,pid2)) ', rank = ' num2str(RankScoreAll(pid,pid2))])


                subplot(2,1,2)
                tmp1 = FingerprintR(1:end,pid);
                tmp1(pid)=[];
                tmp2 = FingerprintR(pid,1:end)';
                tmp2(pid)=[];

                histogram([tmp1;tmp2]);
                hold on
                line([FingerprintR(pid,pid2) FingerprintR(pid,pid2)],get(gca,'ylim'),'color',[1 0 0])
                makepretty

                pause(1)
                delete(tmpfig)
            end

        end
    end

    figure;
    scatter(MatchProbability(:),FingerprintR(:),14,RankScoreAll(:),'filled')
    colormap(cat(1,[0 0 0],winter))
    xlabel('Match Probability')
    ylabel('Cross-correlation fingerprint')
    makepretty
    drawnow

    % Total score larger than threshold
    CandidatePairs = MatchProbability>0.95 & RankScoreAll==1;
    CandidatePairs(tril(true(size(CandidatePairs))))=0;

    close all
end
close all
%% Extract final pairs:
Tbl = array2table(reshape(Predictors,[],size(Predictors,3))); %All parameters
[label, posterior, cost] = predict(BestMdl,Tbl);
MatchProbability = reshape(posterior(:,2),size(Predictors,1),size(Predictors,2));

label = MatchProbability>0.95; % We want to be very confident
[r, c] = find(label==1); %Find matches
Pairs = cat(2,r,c);
Pairs = sort(Pairs,2);
Pairs=unique(Pairs,'rows');
Pairs(Pairs(:,1)==Pairs(:,2),:)=[];
figure; imagesc(label)
colormap(flipud(gray))
xlabel('Unit_i')
ylabel('Unit_j')
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
title('Identified matches')
makepretty

%% inspect probability distributions
figure('name','Parameter Scores');
Edges = [0:0.01:1];
for scid=1:length(Scores2Include)
    eval(['ScoresTmp = ' Scores2Include{scid} ';'])
    ScoresTmp(tril(true(size(ScoresTmp))))=nan;
    subplot(length(Scores2Include),2,(scid-1)*2+1)
    histogram(ScoresTmp(~label),Edges)
    if scid==1
        title('Candidate non-Matches')
    end
    ylabel(Scores2Include{scid})
    makepretty

    subplot(length(Scores2Include),2,scid*2)
    histogram(ScoresTmp(label),Edges)
    if scid==1
        title('Candidate Matches')
    end
    makepretty
end

% 
figure('name','Projected Location Distance to [0 0]')
Dist2Tip = sqrt(nansum(ProjectedLocation.^2,1));
% Dist2TipMatrix = nan(size(CandidatePairs));

Dist2TipMatrix = arrayfun(@(Y) cell2mat(arrayfun(@(X) cat(1,Dist2Tip(X),Dist2Tip(Y)),1:nclus,'Uni',0)),1:nclus,'Uni',0);
Dist2TipMatrix = cat(3,Dist2TipMatrix{:});
Dist2TipMatrix = reshape(Dist2TipMatrix,2,[]);
subplot(1,2,1)
hist3(Dist2TipMatrix(:,~label(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Non-matches')

subplot(1,2,2)
hist3(Dist2TipMatrix(:,label(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Matches')

% Waveform duration
figure('name','WaveDur')
waveformdurationMat = arrayfun(@(Y) cell2mat(arrayfun(@(X) cat(1,waveformduration(X),waveformduration(Y)),1:nclus,'UniformOutput',0)),1:nclus,'UniformOutput',0);
waveformdurationMat = cat(3,waveformdurationMat{:});
subplot(1,2,1)
hist3(waveformdurationMat(:,~label(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Non-matches')

subplot(1,2,2)
hist3(waveformdurationMat(:,label(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Matches')

% SpatialDecaySlope
figure('name','Spatial Decay Slope')
SpatDecMat = arrayfun(@(Y) cell2mat(arrayfun(@(X) cat(1,SpatialDecaySlope(X),SpatialDecaySlope(Y)),1:nclus,'UniformOutput',0)),1:nclus,'UniformOutput',0);
SpatDecMat = cat(3,SpatDecMat{:});
subplot(1,2,1)
hist3(SpatDecMat(:,~label(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Non-matches')

subplot(1,2,2)
hist3(SpatDecMat(:,label(:))')
xlabel('Unit 1')
ylabel('Unit 2')
zlabel('Counts')
title('Candidate Matches')
%%
disp('Recalculate activity correlations')

% Use a bunch of units with high total scores as reference population
Pairs = Pairs(abs(Pairs(:,1)-Pairs(:,2))>=min([nclus-SessionSwitch,]),:);
% Only use every 'unit' once
[val,id1,id2]=unique(Pairs(:,1));
Pairs = Pairs(id1,:);
[val,id1,id2]=unique(Pairs(:,2));
Pairs = Pairs(id1,:);
if npairs>=size(Pairs,1)
    flag = 1;
end

npairs = size(Pairs,1);


% Correlation on first day
srMatches = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),Pairs(:,1),'UniformOutput',0);
srMatches = cat(1,srMatches{:});
srAll = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),1:SessionSwitch-1,'UniformOutput',0);
srAll = cat(1,srAll{:});
SessionCorrelation_Pair1 = corr(srMatches(:,1:size(srMatches,2)./2)',srAll(:,1:size(srMatches,2)./2)');
for pid = 1:size(Pairs,1)
    SessionCorrelation_Pair1(pid,Pairs(pid,1)) = nan;
end


figure('name','Cross-correlation Fingerprints')
subplot(1,3,1)
imagesc(SessionCorrelation_Pair1')
colormap(flipud(gray))
xlabel('Candidate Units to be matched')
ylabel('All units')
title('Day 1')
makepretty

% Correlation on first day
srMatches = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),Pairs(:,2),'UniformOutput',0);
srMatches = cat(1,srMatches{:});
srAll = arrayfun(@(X) histcounts(sp.st(sp.spikeTemplates == AllClusterIDs(Good_Idx(X)) & sp.RecSes == GoodRecSesID(X)),edges),SessionSwitch:nclus,'UniformOutput',0);
srAll = cat(1,srAll{:});
SessionCorrelation_Pair2 = corr(srMatches(:,size(srMatches,2)./2+1:end)',srAll(:,size(srMatches,2)./2+1:end)');
for pid = 1:size(Pairs,1)
    SessionCorrelation_Pair2(pid,Pairs(pid,2)-SessionSwitch+1) = nan;
end

subplot(1,3,2)
imagesc(SessionCorrelation_Pair2')
colormap(flipud(gray))
xlabel('Candidate Units to be matched')
ylabel('All units')
title('Day 2')
makepretty

% Add both together
SessionCorrelations = cat(2,SessionCorrelation_Pair1,SessionCorrelation_Pair2)';

% Correlate 'fingerprints'
FingerprintR = arrayfun(@(X) cell2mat(arrayfun(@(Y) corr(SessionCorrelations(X,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))',SessionCorrelations(Y,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))'),1:nclus,'UniformOutput',0)),1:nclus,'UniformOutput',0);
% FingerprintR = arrayfun(@(X) cell2mat(arrayfun(@(Y) corr(SessionCorrelations(X,~isnan(SessionCorrelations(X,:))&~isnan(SessionCorrelations(Y,:)))',SessionCorrelation_Pair2(Y,~isnan(SessionCorrelation_Pair1(X,:))&~isnan(SessionCorrelation_Pair2(Y,:)))'),1:size(Pairs,1),'UniformOutput',0)),1:size(Pairs,1),'UniformOutput',0);
FingerprintR = cat(1,FingerprintR{:});

% Remove average within and across day correlations
% tmp1 = (FingerprintR(1:SessionSwitch-1,1:SessionSwitch-1));
% FingerprintR(1:SessionSwitch-1,1:SessionSwitch-1) = FingerprintR(1:SessionSwitch-1,1:SessionSwitch-1)-nanmean(tmp1(:));
%
% tmp2 = (FingerprintR(SessionSwitch:end,SessionSwitch:end));
% FingerprintR(SessionSwitch:end,SessionSwitch:end) = FingerprintR(SessionSwitch:end,SessionSwitch:end)-nanmean(tmp2(:));
%
% tmp2 = (FingerprintR(SessionSwitch:end,1:SessionSwitch));
% FingerprintR(SessionSwitch:end,1:SessionSwitch) = FingerprintR(SessionSwitch:end,1:SessionSwitch)-nanmean(tmp2(:));
%
% tmp2 = (FingerprintR(1:SessionSwitch,SessionSwitch:end));
% FingerprintR(1:SessionSwitch,SessionSwitch:end) = FingerprintR(1:SessionSwitch,SessionSwitch:end)-nanmean(tmp2(:));



subplot(1,3,3)
imagesc(FingerprintR)
hold on
line([SessionSwitch SessionSwitch],get(gca,'ylim'),'color',[1 0 0])
line(get(gca,'xlim'),[SessionSwitch SessionSwitch],'color',[1 0 0])
clim([0.5 1])
colormap(flipud(gray))
xlabel('All units across both days')
ylabel('All units across both days')
title('Correlation Fingerprint')
makepretty

%
SigMask = zeros(nclus,nclus);
RankScoreAll = nan(size(SigMask));
for pid=1:nclus
    for pid2 = 1:nclus
        if pid2<SessionSwitch
            tmp1 = FingerprintR(pid,1:SessionSwitch-1);
            addthis=0;
        else
            tmp1 = FingerprintR(pid,SessionSwitch:end);
            addthis = SessionSwitch-1;
        end
        [val,ranktmp] = sort(tmp1,'descend');

        tmp1(pid2-addthis)=[];

        if FingerprintR(pid,pid2)>nanmean(tmp1)+2*nanstd(tmp1)
            SigMask(pid,pid2)=1;

        end
        RankScoreAll(pid,pid2) = find(ranktmp==pid2-addthis);

        if 0%(any(ismember(Pairs(:,1),pid)) & any(ismember(Pairs(:,2),pid2))) || RankScoreAll(pid,pid2)==1
            tmpfig = figure;
            subplot(2,1,1); plot(SessionCorrelations(pid,:)); hold on; plot(SessionCorrelations(pid2,:))
            xlabel('Unit')
            ylabel('Cross-correlation')
            legend('Day1','Day2')
            makepretty
            title(['r=' num2str(FingerprintR(pid,pid2)) ', rank = ' num2str(RankScoreAll(pid,pid2))])


            subplot(2,1,2)
            tmp1 = FingerprintR(1:end,pid);
            tmp1(pid)=[];
            tmp2 = FingerprintR(pid,1:end)';
            tmp2(pid)=[];

            histogram([tmp1;tmp2]);
            hold on
            line([FingerprintR(pid,pid2) FingerprintR(pid,pid2)],get(gca,'ylim'),'color',[1 0 0])
            makepretty

            pause(1)
            delete(tmpfig)
        end

    end
end

% matches = logical(eye(size(Pairs,1)));
% nanidx = isnan(FingerprintR);
% FingerprintR(isnan(FingerprintR))=nanmin(FingerprintR(:)); %Make minimum to not have nans
% [val,rank] = arrayfun(@(X) sort(nanmean([FingerprintR(X,:);FingerprintR(:,X)'],1),'descend'),1:size(Pairs,1),'UniformOutput',0);
% rank = cell2mat(arrayfun(@(X) find(rank{X}==X),1:length(rank),'UniformOutput',0));
% rankscore = zeros(size(Pairs,1),size(Pairs,1));
% rankscore(:,:)=size(Pairs,1)+1;
% rankscore(matches)=rank;
% FingerprintR(nanidx)=nan;
% figure;
% edges = min(FingerprintR(:)):0.1:max(FingerprintR(:));
% histogram(FingerprintR(~matches),edges); hold on; histogram(FingerprintR(matches),edges)
%
% TotalScoreMatches = arrayfun(@(Y) arrayfun(@(X) TotalScore(X,Y),Pairs(:,1),'UniformOutput',0),Pairs(:,2),'UniformOutput',0);
% TotalScoreMatches = cell2mat(cat(2,TotalScoreMatches{:}));
% %
% CrossValFP=FingerprintR;
% CrossValFP(matches)=nan;
% ConfirmedMatch = cell2mat(arrayfun(@(X) FingerprintR(X,X)>nanmax(CrossValFP(X,:)),1:size(Pairs,1),'UniformOutput',0));
% matches = double(matches);
% matches(logical(eye(size(matches))))=ConfirmedMatch+1;

figure;

scatter(MatchProbability(:),FingerprintR(:),14,RankScoreAll(:),'filled')
colormap(cat(1,[0 0 0],winter))
xlabel('MatchProbability')
ylabel('Cross-correlation fingerprint')
makepretty
%% Remove pairs with low ranks
% Pairs(rank>5,:)=[];
% rankscore(rank>5,:)=[];
% rankscore(:,rank>5)=[];
% CorrTraces(rank>5,:)=[];
% FingerprintR(rank>5,:)=[];
% FingerprintR(:,rank>5)=[];
% rank(rank>5)=[];
%% Extract final pairs:
[r, c] = find(label==1); %Find matches
Pairs = cat(2,r,c);
Pairs = sort(Pairs,2);
Pairs=unique(Pairs,'rows');
Pairs(Pairs(:,1)==Pairs(:,2),:)=[];


%% TotalScore Pair versus no pair
SelfScore = MatchProbability(logical(eye(size(MatchProbability))));
scorematches = nan(size(Pairs,1),1); %First being TotalScore, second being TemplateMatch
scoreNoMatches = MatchProbability;
scoreNoMatches(logical(eye(size(MatchProbability))))=nan;

for id = 1:size(Pairs,1)
    scorematches(id,1) = MatchProbability(Pairs(id,1),Pairs(id,2));
    scoreNoMatches(Pairs(id,1),Pairs(id,2),:)=nan;
    scoreNoMatches(Pairs(id,2),Pairs(id,1),:)=nan;
end
ThrsScore = min(MatchProbability(label==1));
figure;
subplot(1,2,1)
histogram(scoreNoMatches(:),[0:0.01:1]); hold on
title('Non Matches')
xlabel('Matching Score')
ylabel('Nr Pairs')
makepretty
subplot(1,2,2)
histogram(SelfScore(:),[0:0.01:1]); hold on
histogram(scorematches(:),[0:0.01:1]); 
line([ThrsScore ThrsScore],get(gca,'ylim'),'color',[1 0 0])

% histogram(scorematches(:,1),[0:0.02:6])
xlabel('Matching Score')
ylabel('Nr Pairs')
legend('Self Score','Matches','Threshold','Location','best')
makepretty

save(fullfile(SaveDir,MiceOpt{midx},'MatchingScores.mat'),'SessionSwitch','GoodRecSesID','AllClusterIDs','Good_Idx','WavformSimilarity','WVCorr','LocDistNorm','waveformdurationcomp','MaxChanloccomp','peaklocscomp','spatialdecaypointscomp','spatialdecaysloopcomp','TemplateMSE','TotalScore','label','MatchProbability')

%% ISI violations (for over splits matching)
ISIViolationsScore = nan(1,size(Pairs,1));
fprintf(1,'Computing functional properties similarity. Progress: %3d%%',0)
for pairid= 1:size(Pairs,1)
    if GoodRecSesID(Pairs(pairid,1)) == GoodRecSesID(Pairs(pairid,2))
        idx1 = sp.spikeTemplates == AllClusterIDs(Good_Idx(Pairs(pairid,1)))&sp.RecSes == GoodRecSesID(Pairs(pairid,1));
        idx2 = sp.spikeTemplates == AllClusterIDs(Good_Idx(Pairs(pairid,2)))&sp.RecSes == GoodRecSesID(Pairs(pairid,2));
        DifScore = diff(sort([sp.st(idx1); sp.st(idx2)]));
        ISIViolationsScore(pairid) = sum(DifScore.*1000<1.5)./length(DifScore);
        fprintf(1,'\b\b\b\b%3.0f%%',pairid/size(Pairs,1)*100)
       
    end
end
fprintf('\n')
disp(['Removing ' num2str(sum(ISIViolationsScore>0.05)) ' matched oversplits, as merging them will violate ISI >5% of the time'])
Pairs(ISIViolationsScore>0.05,:)=[];

%% Figures
% Pairs = Pairs(any(ismember(Pairs,[8,68,47,106]),2),:);
AllClusterIDs(Good_Idx(Pairs))
for pairid=1:size(Pairs,1)
    uid = Pairs(pairid,1);
    uid2 = Pairs(pairid,2);

    pathparts = strsplit(AllRawPaths{GoodRecSesID(uid)},'\');
    rawdatapath = dir(fullfile('\\',pathparts{1:end-1}));
    if isempty(rawdatapath)
        rawdatapath = dir(fullfile(pathparts{1:end-1}));
    end

    % Load raw data
    SM1=load(fullfile(rawdatapath(1).folder,'RawWaveforms',['Unit' num2str(num2str(AllClusterIDs(Good_Idx(uid)))) '_RawSpikes.mat']));
    SM1 = SM1.spikeMap; %Average across these channels

    pathparts = strsplit(AllRawPaths{GoodRecSesID(uid2)},'\');
    rawdatapath = dir(fullfile('\\',pathparts{1:end-1}));
    if isempty(rawdatapath)
        rawdatapath = dir(fullfile(pathparts{1:end-1}));
    end

    SM2=load(fullfile(rawdatapath(1).folder,'RawWaveforms',['Unit' num2str(num2str(AllClusterIDs(Good_Idx(uid2)))) '_RawSpikes.mat']));
    SM2 = SM2.spikeMap; %Average across these channels

    tmpfig = figure;
    subplot(3,3,[1,4])
    Locs = channelpos(ChanIdx{uid},:);
    for id = 1:length(Locs)
        plot(Locs(id,1)*5+[1:size(SM1,1)],Locs(id,2)*10+nanmean(SM1(:,id,:),3),'b-','LineWidth',1)
        hold on
    end
    plot(ProjectedLocation(1,uid)*5+[1:size(SM1,1)],ProjectedLocation(2,uid)*10+ProjectedWaveform(:,uid),'b--','LineWidth',2)


    Locs = channelpos(ChanIdx{uid2},:);
    for id = 1:length(Locs)
        plot(Locs(id,1)*5+[1:size(SM2,1)],Locs(id,2)*10+nanmean(SM2(:,id,:),3),'r-','LineWidth',1)
        hold on
    end
    plot(ProjectedLocation(1,uid2)*5+[1:size(SM1,1)],ProjectedLocation(2,uid2)*10+ProjectedWaveform(:,uid2),'r--','LineWidth',2)

    makepretty
    set(gca,'xticklabel',arrayfun(@(X) num2str(X./5),cellfun(@(X) str2num(X),get(gca,'xticklabel')),'UniformOutput',0))
    set(gca,'yticklabel',arrayfun(@(X) num2str(X./10),cellfun(@(X) str2num(X),get(gca,'yticklabel')),'UniformOutput',0))
    xlabel('Xpos (um)')
    ylabel('Ypos (um)')
    title(['unit' num2str(AllClusterIDs(Good_Idx(uid))) ' versus unit' num2str(AllClusterIDs(Good_Idx(uid2))) ', ' 'RecordingDay ' num2str(GoodRecSesID(uid)) ' versus ' num2str(GoodRecSesID(uid2)) ', Probability=' num2str(round(MatchProbability(uid,uid2).*100)) '%'])


    subplot(3,3,[2])
    plot(AllTemplates(uid,:));
    hold on
    plot(AllTemplates(uid2,:));
    legend(num2str(AllClusterIDs(Good_Idx(uid))),num2str(AllClusterIDs(Good_Idx(uid2))),'Location','best');
    makepretty
    title(['Template r='  num2str(round(TemplateMSE(uid,uid2)*100)/100)])

    subplot(3,3,5)
    plot(channelpos(:,1),channelpos(:,2),'k.')
    hold on
    h(1)=plot(channelpos(MaxChan(uid),1),channelpos(MaxChan(uid),2),'b.','MarkerSize',15);
    h(2) = plot(channelpos(MaxChan(uid2),1),channelpos(MaxChan(uid2),2),'r.','MarkerSize',15);
    xlabel('X position')
    ylabel('um from tip')
    makepretty
    title(['Chan ' num2str(MaxChan(uid)) ' versus ' num2str(MaxChan(uid2))])

    subplot(3,3,3)
    hold on
    SM1 = squeeze(nanmean(SM1,2));
    SM2 = squeeze(nanmean(SM2,2));
    h(1)=plot(nanmean(SM1(:,1:2:end),2),'b-');
    h(2)=plot(nanmean(SM1(:,2:2:end),2),'b--');
    h(3)=plot(nanmean(SM2(:,1:2:end),2),'r-');
    h(4)=plot(nanmean(SM2(:,2:2:end),2),'r--');
    makepretty
    title(['Waveform Similarity Score =' num2str(round(WavformSimilarity(uid,uid2)*1000)./1000)])


    % Scatter spikes of each unit
    subplot(3,3,6)
    idx1=find(sp.spikeTemplates == AllClusterIDs(Good_Idx(uid)) & sp.RecSes == GoodRecSesID(uid));
    scatter(sp.st(idx1)./60,sp.spikeAmps(idx1),4,[0 0 1],'filled')
    hold on
    idx2=find(sp.spikeTemplates == AllClusterIDs(Good_Idx(uid2)) &  sp.RecSes == GoodRecSesID(uid2));
    scatter(sp.st(idx2)./60,-sp.spikeAmps(idx2),4,[1 0 0],'filled')
    xlabel('Time (min)')
    ylabel('Abs(Amplitude)')
    title(['Amplitude distribution'])
    xlims = get(gca,'xlim');
    ylims = max(abs(get(gca,'ylim')));
    % Other axis
    [h1,edges,binsz]=histcounts(sp.spikeAmps(idx1));
    %Normalize between 0 and 1
    h1 = ((h1-nanmin(h1))./(nanmax(h1)-nanmin(h1)))*10+xlims(2)+10;
    plot(h1,edges(1:end-1),'b-');
    [h2,edges,binsz]=histcounts(sp.spikeAmps(idx2));
    %Normalize between 0 and 1
    h2 = ((h2-nanmin(h2))./(nanmax(h2)-nanmin(h2)))*10+xlims(2)+10;
    plot(h2,-edges(1:end-1),'r-');
    ylabel('Amplitude')
    ylim([-ylims ylims])

    makepretty


    % compute ACG
    [ccg, ~] = CCGBz([double(sp.st(idx1)); double(sp.st(idx1))], [ones(size(sp.st(idx1), 1), 1); ...
        ones(size(sp.st(idx1), 1), 1) * 2], 'binSize', param.ACGbinSize, 'duration', param.ACGduration, 'norm', 'rate'); %function
    ACG = ccg(:, 1, 1);
    [ccg, ~] = CCGBz([double(sp.st(idx2)); double(sp.st(idx2))], [ones(size(sp.st(idx2), 1), 1); ...
        ones(size(sp.st(idx2), 1), 1) * 2], 'binSize', param.ACGbinSize, 'duration', param.ACGduration, 'norm', 'rate'); %function
    ACG2 = ccg(:, 1, 1);
    [ccg, ~] = CCGBz([double(sp.st([idx1;idx2])); double(sp.st([idx1;idx2]))], [ones(size(sp.st([idx1;idx2]), 1), 1); ...
        ones(size(sp.st([idx1;idx2]), 1), 1) * 2], 'binSize', param.ACGbinSize, 'duration', param.ACGduration, 'norm', 'rate'); %function

    subplot(3,3,7); plot(ACG,'b');
    hold on
    plot(ACG2,'r')
    title(['AutoCorrelogram'])
    makepretty
    subplot(3,3,8)

    if exist('NatImgCorr','var')
        if GoodRecSesID(uid)==1 % Recording day 1
            tmp1 = squeeze(D0(OriginalClusID(Good_Idx(uid))+1,:,:));
        else % Recordingday 2
            tmp1 = squeeze(D1(OriginalClusID(Good_Idx(uid))+1,:,:));
        end
        if GoodRecSesID(uid2)==1 % Recording day 1
            tmp2 = squeeze(D0(OriginalClusID(Good_Idx(uid2))+1,:,:));
        else % Recordingday 2
            tmp2 = squeeze(D1(OriginalClusID(Good_Idx(uid2))+1,:,:));
        end

        plot(nanmean(tmp1,1),'b-');
        hold on
        plot(nanmean(tmp2,1),'r-');
        xlabel('Stimulus')
        ylabel('NrSpks')
        makepretty


        if AllClusterIDs(Good_Idx(uid))  == AllClusterIDs(Good_Idx(uid2))
            if ismember(Good_Idx(uid),Good_ClusUnTracked)
                title(['Visual: Untracked, r=' num2str(round(NatImgCorr(pairid,pairid)*100)/100)])
            elseif ismember(Good_Idx(uid),Good_ClusTracked)
                title(['Visual: Tracked, r=' num2str(round(NatImgCorr(pairid,pairid)*100)/100)])
            else
                title(['Visual: Unknown, r=' num2str(round(NatImgCorr(pairid,pairid)*100)/100)])
            end
        else
            title(['Visual: Unknown, r=' num2str(round(NatImgCorr(pairid,pairid)*100)/100)])
        end
    else
        isitot = diff(sort([sp.st(idx1); sp.st(idx2)]));
        histogram(isitot,'FaceColor',[0 0 0])
        hold on
        line([1.5/1000 1.5/1000],get(gca,'ylim'),'color',[1 0 0],'LineStyle','--')
        title([num2str(round(sum(isitot*1000<1.5)./length(isitot)*1000)/10) '% ISI violations']); %The higher the worse (subtract this percentage from the Total score)
        xlabel('ISI (ms)')
        ylabel('Nr. Spikes')
        makepretty
    end

    subplot(3,3,9)
    
    plot(SessionCorrelations(Pairs(pairid,1),:),'b-'); hold on; plot(SessionCorrelations(Pairs(pairid,2),:),'r-')
    xlabel('Unit')
    ylabel('Cross-correlation')   
    title(['Fingerprint r=' num2str(round(FingerprintR(Pairs(pairid,1),Pairs(pairid,2))*100)/100) ', rank=' num2str(RankScoreAll(Pairs(pairid,1),Pairs(pairid,2)))])
    makepretty

    disp(['UniqueID ' num2str(AllClusterIDs(Good_Idx(uid))) ' vs ' num2str(AllClusterIDs(Good_Idx(uid2)))])
    disp(['Peakchan ' num2str(MaxChan(uid)) ' versus ' num2str(MaxChan(uid2))])
    disp(['RecordingDay ' num2str(GoodRecSesID(uid)) ' versus ' num2str(GoodRecSesID(uid2))])
    disp(['TemplateR ' num2str(round(TemplateMSE(uid,uid2)*100)/100) ',  Waveform similarity ' num2str(round(WavformSimilarity(uid,uid2)*100)/100)])

    drawnow
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    saveas(gcf,fullfile(SaveDir,MiceOpt{midx},['ClusID' num2str(AllClusterIDs(Good_Idx(uid))) 'vs' num2str(AllClusterIDs(Good_Idx(uid2))) '.fig']))
    saveas(gcf,fullfile(SaveDir,MiceOpt{midx},['ClusID' num2str(AllClusterIDs(Good_Idx(uid))) 'vs' num2str(AllClusterIDs(Good_Idx(uid2))) '.bmp']))

end

if 0
% look for natural images data
% AL data from Kush:
D0 = readNPY('H:\Anna_TMP\image_analysis\responses\day_0\template_responses.npy');
D1 = readNPY('H:\Anna_TMP\image_analysis\responses\day_1\template_responses.npy');
% matrix: number of spikes from 0 to 0.7seconds after stimulus onset: n_templates X n_reps X n_Images
% Get rid of the units not currently looked at (We have only shank 0 here)
D0(1:end-length(unique(AllClusterIDs)),:,:)=[];
D1(1:end-length(unique(AllClusterIDs)),:,:)=[];
NatImgCorr = nan(nclus,nclus);
nrep = size(D0,2);
for uid=1:nclus
    uid
    if GoodRecSesID(uid)==1 % Recording day 1
        tmp1 = squeeze(D0(OriginalClusID(Good_Idx(uid))+1,:,:));
    else % Recordingday 2
        tmp1 = squeeze(D1(OriginalClusID(Good_Idx(uid))+1,:,:));
    end
    parfor uid2 = uid:nclus
        if GoodRecSesID(uid2)==1 % Recording day 1
            tmp2 = squeeze(D0(OriginalClusID(Good_Idx(uid2))+1,:,:));
        else % Recordingday 2
            tmp2 = squeeze(D1(OriginalClusID(Good_Idx(uid2))+1,:,:));
        end
% 
%         figure; subplot(2,2,1); imagesc(tmp1); title(['Day ' num2str(GoodRecSesID(uid)) ', Unit ' num2str(OriginalClusID(Good_Idx(uid)))])
%         colormap gray
%         colorbar; xlabel('Condition'); ylabel('Repeat')
%         hold on; subplot(2,2,2); imagesc(tmp2); title(['Day ' num2str(GoodRecSesID(uid2)) ', Unit ' num2str(OriginalClusID(Good_Idx(uid2)))])
%         colormap gray
%         colorbar; xlabel('Condition'); ylabel('Repeat')
%         subplot(2,2,3); hold on

        % Is the unit's response predictable?
        tmpcor = nan(1,nrep);
        for cv = 1:nrep
            % define training and test
            trainidx = circshift(1:nrep,-(cv-1));
            testidx = trainidx(1);
            trainidx(1)=[];

            % Define response:
            train = nanmean(tmp1(trainidx,:),1);
            test = tmp2(testidx,:);          

            % Between error
            tmpcor(1,cv) = corr(test',train');
%             scatter(train,test,'filled')
            
%             plot(train);

        end
        NatImgCorr(uid,uid2) = nanmean(nanmean(tmpcor,2));

%         xlabel('train')
%         ylabel('test')
%         title(['Average Correlation ' num2str(round(NatImgCorr(uid,uid2)*100)/100)])
%        lims = max(cat(1,get(gca,'xlim'),get(gca,'ylim')),[],1);
%         set(gca,'xlim',lims,'ylim',lims)
%         makepretty

    end
end
% Mirror these
for uid2 = 1:nclus
    for uid=uid2+1:nclus
        NatImgCorr(uid,uid2)=NatImgCorr(uid2,uid);
    end
end
NatImgCorr = arrayfun(@(Y) arrayfun(@(X) NatImgCorr(Pairs(X,1),Pairs(Y,2)),1:size(Pairs,1),'UniformOutput',0),1:size(Pairs,1),'UniformOutput',0)
NatImgCorr = cell2mat(cat(1,NatImgCorr{:}));

% Kush's verdict:
Good_ClusTracked = readNPY('H:\Anna_TMP\image_analysis\cluster_ids\good_clusters_tracked.npy'); % this is an index, 0 indexed so plus 1
Good_ClusUnTracked = readNPY('H:\Anna_TMP\image_analysis\cluster_ids\good_clusters_untracked.npy') % this is an index, 0 indexed so plus 1

Good_ClusTracked(Good_ClusTracked>max(AllClusterIDs))=[]; % 
Good_ClusUnTracked(Good_ClusUnTracked>max(AllClusterIDs)) = [];

NotIncluded = [];
TSGoodGr = nan(1,length(Good_ClusTracked));
for uid = 1:length(Good_ClusTracked)
    idx = find(AllClusterIDs(Good_Idx) == Good_ClusTracked(uid));
    if length(idx)==2
        TSGoodGr(uid) = TotalScore(idx(1),idx(2));
    else
        NotIncluded = [NotIncluded  Good_ClusTracked(uid)];
    end
end
TSBadGr = nan(1,length(Good_ClusUnTracked));
for uid = 1:length(Good_ClusUnTracked)
    idx = find(AllClusterIDs(Good_Idx) == Good_ClusUnTracked(uid));
    if length(idx)==2
        TSBadGr(uid) = TotalScore(idx(1),idx(2));
    else
        NotIncluded = [NotIncluded  Good_ClusUnTracked(uid)];
    end
end
figure; histogram(TSBadGr,[5:0.05:6]); hold on; histogram(TSGoodGr,[5:0.05:6])
xlabel('Total Score')
ylabel('Nr. Matches')
legend({'Not tracked','Tracked'})
makepretty
%Tracked?
Tracked = zeros(nclus,nclus);
for uid=1:nclus
    parfor uid2 = 1:nclus
        if uid==uid2
            Tracked(uid,uid2)=0;
        elseif AllClusterIDs(Good_Idx(uid))  == AllClusterIDs(Good_Idx(uid2))
            if ismember(Good_Idx(uid),Good_ClusUnTracked) ||  ismember(Good_Idx(uid2),Good_ClusUnTracked) 
                Tracked(uid,uid2)=-1;
            elseif ismember(Good_Idx(uid),Good_ClusTracked)||  ismember(Good_Idx(uid2),Good_ClusTracked) 
                Tracked(uid,uid2)=1;
            else
                Tracked(uid,uid2) = 0.5;
            end
        end
    end
end
end

