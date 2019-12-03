function simOutput = positronSimulation(dr, fns, nametag, simOpts)

if nargin<3
    nametag = '';
end
%clear the parallel pool
%delete(gcp('nocreate'))

if ~nargin || isempty(dr)
    [fns, dr] = uigetfile('*.mat', 'Select your data Chunks', 'multiselect', 'on');
    if ~iscell(fns)
        fns = {fns};
    end
end

%options
opts.discard = 0; %Time in seconds to discard from beginning due to LED being off (discarded from every datasetblock)
opts.doCrossVal = false;
opts.doGlobalSubtract = false;

opts.contextSize = 50; % 65; %number of pixels surrounding the ROI to use as context
opts.censorSize = 19; %number of pixels surrounding the ROI to censor from the background PCA; roughly the spatial scale of scattered/dendritic neural signals, in pixels.
opts.nPC_bg = 1; %number of principle components used for background subtraction
opts.tau_lp = 3; %time window for lowpass filter (seconds); signals slower than this will be ignored
opts.tau_pred = 1; % time window in seconds for high pass filtering to make predictor for regression
opts.sigmas = [1 1.5 2]; %spatial smoothing radius imposed on spatial filter;
opts.nIter = 5; %number of iterations alternating between estimating temporal and spatial filters.
opts.localAlign = false;
opts.highPassRegression = false;

%simulation options
opts.S.cellRadius = 3.5; %cell radius in pixels
opts.S.nSpikes = 100; %approx # of spikes in recording
opts.S.ISI = 0.01; %minimum inter-spike interval, in seconds.
opts.S.fixLocs = false;
opts.S.nCells = 10; %# of cells
opts.S.dist = 7; % minimum distance between cells
opts.S.stCorr = 0; %subthreshold correlation NOT IMPLEMENTED
opts.S.simSpike = [0,-0.00610252301264466,0.0153324173640442,0.0394596567723412,0.0550890774222366,0.0855483409375236,0.109121613111175,0.159347286748449,0.272280693885892,0.454264679126972,1,0.678615836282434,0.441859762420943,0.232255883375386,0.0936575998651618,-0.0134953901062200,-0.0507224917483669,-0.0559150686307210,-0.0532159906094771,-0.0212346446514174,0]; %positive-going spike
opts.S.spikeSize = 3; %spike amplitude in multiples of subthreshold amplitude
opts.S.spikeAmp = 0.10; %Spike amplitude, percent, opsin only, no background. Theoretical max is 100%
opts.S.bHi = 10^3.2;%cell brightness in brightest state; kept constant across sensitivities
opts.S.bLo = opts.S.bHi./(1 + opts.S.spikeAmp); %cell brightness in dimmest state
opts.S.tBleach = 1e6; %bleaching time constant, in frames
opts.S.epsp = false; %NOT IMPLEMENTED

%parameter matrix
if nargin<4
    SAs = 0.1;
    NCs = 25;
else
    SAs = simOpts.SAs; NCs = simOpts.NCs;
end

[SAv, NCv] = ndgrid(SAs,NCs);
opts.S.nRep = 1; 
SAv = repmat(SAv(:), opts.S.nRep,1);
NCv = repmat(NCv(:), opts.S.nRep,1);
opts.S.nIter =length(NCv);

for fn_ix = 1:length(fns)
    simOutput =[];
    disp(['Loading data batch: ' fns{fn_ix}])
    struct = load([dr filesep fns{fn_ix}], 'data', 'sampleRate');
    data0 = struct.data;
    imavg = mean(data0(:,:, round (linspace(101, size(data0,3), 1000))), 3);
    sampleRate = struct.sampleRate;
    opts.windowLength =  sampleRate*0.02; %window length for spike templates
    nFrames = size(data0,3);
    bleaching = (0.5 + exp(-(0:nFrames-1)./opts.S.tBleach))./1.5;
        
    for simIter = 1:opts.S.nIter
        disp(['iterations left: ' int2str(simIter)])
        
        rng(sum(cast(fns{1}, 'uint32'))+simIter); %fix random number generator
        
        simOutput(simIter,1).spikeAmp = SAv(simIter);
        simOutput(simIter,1).nCells = NCv(simIter);
        simOutput(simIter,1).imavg = imavg;
        opts.S.nCells = simOutput(simIter,1).nCells;
        opts.S.spikeAmp = simOutput(simIter,1).spikeAmp;
        
        %ADD SIMULATED NEURON AND GENERATE ROI
        %simulate subthreshold
        for cNum = 1:opts.S.nCells
            subThresh(:,cNum) = imgaussfilt(max(-2,randn(1, nFrames)), 0.02*sampleRate);
            subThresh(:,cNum) = subThresh(:,cNum)./prctile(subThresh(:,cNum),95);  subThresh(:,cNum) = subThresh(:,cNum)-prctile(subThresh(:,cNum),5);
        end

        for cNum = 1:opts.S.nCells
            vTimes = cumsum(randi(ceil([opts.S.ISI*sampleRate (2*opts.S.ISI*sampleRate)-1]), 1, length(subThresh))); vTimes = vTimes(find(vTimes>length(opts.S.simSpike),1,'first'):find(vTimes<(nFrames-length(opts.S.simSpike)),1,'last'));
            ST{cNum} = randsample(vTimes, opts.S.nSpikes, true, max(subThresh(vTimes),0));
            ST{cNum} = unique(ST{cNum});
            t2 = zeros(1, size(subThresh,1)); t2(ST{cNum}) = 1; t2 = conv(t2, opts.S.simSpike, 'same');
            
            %generate traces, which range between 1 and 1+spikeAmp, with
            %the correct sign. This will be multiplied by the minimum
            %brightness to get signals that range between Fmin and Fmax
            tmpTrace=(subThresh(:,cNum)./opts.S.spikeSize + t2');
            tmpTrace = tmpTrace-median(tmpTrace);
            simTraceNeg(:,cNum) = (1+opts.S.spikeAmp) - (tmpTrace.*opts.S.spikeAmp);
            simTracePos(:,cNum) = 1 + (tmpTrace.*opts.S.spikeAmp);
        end
        %simulate location
        imcells = imgaussfilt(imtophat(imavg, ones(round(3*opts.S.cellRadius))), opts.S.cellRadius);
        validlocs = imcells<max(imcells(:))/4;
        validlocs([1:2*opts.S.cellRadius+2 end-2*opts.S.cellRadius-1:end], :) = false; validlocs(:,[1:2*opts.S.cellRadius+2 end-2*opts.S.cellRadius-2:end]) = false;
        validlocs(imavg<prctile(imavg(:), 20) | imcells<prctile(imcells(:), 15)) = false;
        rng(simIter);
        SE =strel('disk', ceil(2*opts.S.cellRadius), 4);
        for lx = opts.S.nCells:-1:1
            [locR(lx),locC(lx)] = ind2sub(size(imavg), randsample(find(validlocs(:)),1));
            mask = true(size(validlocs)); mask(locR(lx),locC(lx)) = false; mask=imerode(mask, SE);
            validlocs = validlocs & mask;
        end

        %neuron shape
        neuronIM = zeros(4*opts.S.cellRadius+1);
        center = neuronIM;
        center(ceil(end/2), ceil(end/2)) = 1;
        neuronIM(2:ceil(end/2), ceil(end/2)) = 1;
        neuronIM(imdilate(center, strel('disk', floor(opts.S.cellRadius),0))>0) = 1;
        neuronIM(imdilate(center, strel('disk', floor(opts.S.cellRadius-1),0))>0) = 0.4;
        ROIs = false([size(imavg) opts.S.nCells]);
        for cellN = opts.S.nCells:-1:1
            Ntmp = imrotate(neuronIM, 360*rand, 'bilinear');
            Ntmp2 = zeros(size(imavg));
            Ntmp2(locR(cellN)+ ([1:size(Ntmp,1)]-(min(ceil(size(Ntmp,1)/2), locR(cellN)))), locC(cellN)+ ([1:size(Ntmp,2)]-min(ceil(size(Ntmp,2)/2), locC(cellN)))) = Ntmp;
            Ntmp2 = 1/3*imgaussfilt(Ntmp2, 0.8) + 2/3*imgaussfilt(Ntmp2, 20);
            IM2(:,:, cellN) = Ntmp2(1:size(imavg,1), 1:size(imavg,2));
            
            %create ROI
            SProi = false(size(imavg)); SProi(locR(cellN),locC(cellN)) = true; SProi = imdilate(SProi, strel('disk', ceil(opts.S.cellRadius+0.5)));
            ROIs(:,:,cellN) = SProi;
        end
        
        %save ground truth data
        simOutput(simIter,1).gt.ST = ST; %spike times
        simOutput(simIter,1).gt.IM2 = IM2; %spatial footprints
        simOutput(simIter,1).gt.bleaching = bleaching;
        simOutput(simIter,1).gt.trace = cat(3, simTraceNeg,simTracePos);
        
        for sigSign = [1,2] %1 voltron, 2 positron
            if sigSign==1
                simTrace = max(0, simTraceNeg);
                multiplier = -1;
            else
                simTrace = max(0,simTracePos);
                multiplier = 1;
            end
            
%             figure, 
%             ax1 = subplot(2,1,1); plot(simTraceNeg(:,1), 'r');
%             hold on, plot([1 40000], [1 1], ':k'); hold on, plot([1 40000], (1+simOutput(simIter,1).spikeAmp).*[1 1], ':k'); 
%             set(gca, 'ylim', [0,2]);
%             ax2 = subplot(2,1,2); plot(simTracePos(:,1), 'b')
%             hold on, plot([1 40000], [1 1], ':k'); hold on, plot([1 40000], (1+simOutput(simIter,1).spikeAmp).*[1 1], ':k'); 
%             set(gca, 'ylim', [0,2]);
%             linkaxes([ax1 ax2])
            
            disp(['Simulating brightness:' int2str(opts.S.bLo)])
            %add neurons to data
            dSim = zeros(size(data0), 'single');
            for cellN = 1:opts.S.nCells
                dSim =  dSim + IM2(:,:,cellN).*reshape(bleaching.*simTrace(:,cellN)'.*opts.S.bLo, 1, 1, []);
                %for every cell we place, put down XXX out of focus cells with scrambled activity
                bb = sum(Ntmp2(:));
                for oof = 1:6
                    IM_oof = zeros(size(IM2,1), size(IM2,2));
                    IM_oof(ceil(rand*numel(IM_oof))) = bb;
                    IM_oof = imgaussfilt(IM_oof, 12+10*rand, 'FilterSize', [111,111], 'Padding', 0);
                    oof_trace = simTrace(randperm(size(simTrace,1)),cellN);
                    dSim = dSim + IM_oof.*reshape(bleaching.*oof_trace'.*opts.S.bLo, 1, 1, []);
                end
            end
            ENF = 2; %excess noise factor
            dSim = dSim + ENF.*randn(size(dSim)).*sqrt(max(10,dSim));
            dataAll = data0 + max(0,dSim);
            
            simOutput(simIter,sigSign).simFrame = dataAll(:,:,1);
            figure, imagesc(dataAll(:,:,1)); axis image; drawnow
            
            %Compute global PCs with ROIs masked out
            if opts.doGlobalSubtract
                mask = ~imdilate(any(ROIs,3), strel('disk', opts.censorSize));
                data = reshape(dataAll, [], size(dataAll, 3));
                data = double(data(mask(:),:));
                disp('Performing highpass filtering');
                tic
                data = highpassVideo(data', 1/opts.tau_lp, sampleRate)'; %takes ~2-3 minutes
                toc
                disp('Performing PCA...')
                tic
                [~,~,Vg_hp] = svds(data, opts.nPC_bg); %takes ~2-3 minutes
                toc
                Vg_pred = highpassVideo(Vg_hp, 1/opts.tau_pred, sampleRate); %filter Vg
            end
            
            for cellN = 1:size(ROIs,3)
                disp(['Processing cell:' int2str(cellN)]);
                tic
                bw = ROIs(:,:,cellN);
                
                %extract relevant region
                bwexp = imdilate(bw, ones(opts.contextSize));
                Xinds = find(any(bwexp,1),1,'first'):find(any(bwexp,1),1,'last');
                Yinds= find(any(bwexp,2),1,'first'):find(any(bwexp,2),1,'last');
                bw = bw(Yinds,Xinds);
                notbw = ~imdilate(bw, strel('disk', opts.censorSize));
                data = dataAll(Yinds, Xinds, :);
                bw = logical(bw);
                notbw = logical(notbw);
                
                ref = median(double(data(:,:,1:500)),3);
                
                output.meanIM = mean(data,3); figure, imagesc(output.meanIM);
                data = reshape(data, [], size(data,3));
                
                data = double(data);
                data = double(data-mean(data,2)); %mean subtract
                data = data-mean(data,2); %do it again because of numeric issues
                
                %remove low frequency components
                data_hp = highpassVideo(data', 1/opts.tau_lp, sampleRate)';
                data_lp = data-data_hp;
                
                if opts.highPassRegression
                    data_pred =  highpassVideo(data', 1/opts.tau_pred, sampleRate)';
                else
                    data_pred = data_hp;
                end
               
                
                t = nanmean(double(data_hp(bw(:),:)),1); %initial trace is just average of ROI
                t = t-mean(t);
                output.t = t;
                
%                 %remove any variance in trace that can be predicted from the background PCs
                    if opts.doGlobalSubtract 
                        b = regress(t', Vg_hp);
                        t = t'-(Vg_hp*b);
                    else
                        [~,~,Vb] = svds(double(data_hp(notbw(:),:)), opts.nPC_bg);
                        b = ridge(t',Vb,1); %b = regress(t', Vb);
                        t = (t'-Vb*b); %initial trace
                        %t=t';
                    end

                %Initial spike estimate
                [Xspikes, spikeTimes, guessData, output.rawROI.falsePosRate, output.rawROI.detectionRate, output.rawROI.templates, low_spk] = denoiseSpikes(multiplier.*t', opts.windowLength, sampleRate,true, 100);
                Xspikes = multiplier.*Xspikes; %check on guessData?
                output.rawROI.X = t';
                output.rawROI.Xspikes = Xspikes;
                output.rawROI.spikeTimes = spikeTimes;
                output.rawROI.spatialFilter = bw;
                output.rawROI.X = output.rawROI.X.*(mean(t(output.rawROI.spikeTimes))/mean(output.rawROI.X(output.rawROI.spikeTimes)));%correct shrinkage
                
                %prebuild the regression matrix
                pred = [ones(1,size(data_pred,2)); reshape(imgaussfilt(reshape(data_pred, [size(ref) size(data,2)]), 1.5), size(data))]'; %generate a predictor for ridge regression
                
                % To do: if not enough spikes, take spatial filter from previous block
                
                % Cross-validation of regularized regression parameters
                lambdamax = norm(pred(2:end,:),'fro').^2;
                lambdas = lambdamax*logspace(-4,-2,3); %if you want multiple values of lambda
                I0 = eye(size(pred,2)); I0(1)=0;
                
                if opts.doCrossVal
                    num_batches = 3;
                    batchsize = floor(size(data,2)/num_batches);
                    for batch = 1:num_batches
                        disp(['crossvalidating lambda, batch ' int2str(batch) ' of ' int2str(num_batches)])
                        select = false(size(guessData));
                        select((batch-1)*batchsize + (1:batchsize)) = true;
                        for s_ix = 1:length(opts.sigmas)
                            pred = [ones(1,size(data_pred,2)); reshape(imgaussfilt(reshape(data_pred, [size(ref) size(data_pred,2)]), opts.sigmas(s_ix)), size(data_pred))]';
                            for l_ix = 1:length(lambdas)
                                kk2= (pred(~select, :)'*pred(~select, :)+lambdas(l_ix)*I0)\pred(~select, :)';
                                weights = kk2*(guessData(~select))'; %regression
                                corrs2(l_ix, s_ix, batch) = corr(pred(select, :)*weights, guessData(select)');  %% ok<AGROW>
                            end
                        end
                    end
                    [l_max, s_max] = find(nanmean(corrs2, 3) == nanmax(nanmax(nanmean(corrs2, 3))));
                    opts.lambda = lambdas(l_max);
                    opts.lambda_ix = l_max;
                    opts.sigma = opts.sigmas(s_max);
                    if isempty(s_max)
                        disp('a cell had no spikes.... continuing')
                        continue
                    end
                else %fix the values:
                    s_max = 2;
                    l_max = 3;
                    opts.lambda = lambdas(l_max);
                    opts.sigma = opts.sigmas(s_max);
                    opts.lambda_ix = l_max;
                end
                
                selectPred = true(1,size(data,2));
                if opts.highPassRegression
                    selectPred([1:(sampleRate/2+1) (end-sampleRate/2):end]) = false; %discard data at edges to avoid any filtering artefacts; optional
                end
                    
                pred = [ones(1,size(data_pred,2)); reshape(imgaussfilt(reshape(data_pred, [size(ref) size(data_pred,2)]), opts.sigmas(s_max)), size(data_pred))]';
                recon = [ones(1,size(data_hp,2)); reshape(imgaussfilt(reshape(data_hp, [size(ref) size(data_hp,2)]), opts.sigmas(s_max)), size(data_hp))]';
                kk = (pred(selectPred,:)'*pred(selectPred,:) +lambdas(l_max)*I0)\pred(selectPred,:)';
                  
                for iter = 1:opts.nIter
                    doPlot = false;
                    if iter==opts.nIter
                        doPlot = true;
                    end
                    
                    disp('Identifying spatial filters') %identify spatial filters with regularized regression
                    gD = guessData(selectPred); select = gD~=0;
                    weights = kk(:,select)*gD(select)'; %regression
                    X = double(recon*weights)';
                    X = X-mean(X);
                    
                    spatialFilter = imgaussfilt(reshape(weights(2:end), size(ref)),opts.sigmas(s_max));
                    %remove background contamination; spike pursuit alone does not do this per se

                    if opts.doGlobalSubtract && iter==opts.nIter
                        b = regress(X', Vg_hp);
                        X = X-(Vg_hp*b)';
                        
                        b = regress(X', Vg_pred);
                        X = X-(Vg_pred*b)';
                        output.Vg = Vg_hp; %global background components
                        output.b = b; %weights of global background components that were subtracted to produce y
                    else
                        b = regress(X', Vb);
                        X = X-(Vb*b)';
                    end
                    X = X.*(mean(t(spikeTimes))/mean(X(spikeTimes)));%correct shrinkage
                    
                    %generate the new trace and the new denoised trace
                    clipSpikes = 100;
                    [Xspikes, spikeTimes, guessData, falsePosRate, detectionRate, templates, ~] = denoiseSpikes(multiplier.*X, opts.windowLength, sampleRate,doPlot, clipSpikes);
                    drawnow;
                end
                
                %ensure that the maximum of the spatial filter is within the ROI
                IMcorr = corr(multiplier.*guessData', pred(:, 2:end));
                maxCorrInROI = max(IMcorr(bw(:)));
                if any(IMcorr(~bw(:))>maxCorrInROI)
                    output.passedLocalityTest = false;
                else
                    output.passedLocalityTest = true;
                end
                
                %compute SNR
                selectSpikes = false(length(Xspikes),1); selectSpikes(spikeTimes) = true;
                signal = mean(Xspikes(selectSpikes));
                noise = std(Xspikes(~selectSpikes));
                snr = signal/noise;
                output.snr = snr;
                
                %output
                output.y = X;
                output.yFilt = multiplier.*Xspikes;
                output.ROI = [Xinds([1 end])'  Yinds([1 end])'];
                output.ROIbw = bw;
                output.spatialFilter = spatialFilter;
                output.falsePosRate = falsePosRate;
                output.detectionRate = detectionRate;
                output.templates = templates;
                output.spikeTimes = spikeTimes;
                output.opts = opts;
                output.F0 = nanmean(double(data_lp(bw(:),:))+output.meanIM(bw(:)),1);
                output.dFF = X(:)./output.F0(:);
                output.rawROI.dFF = output.rawROI.X(:)./output.F0(:);
                %output.Vb = Vb; %local background components
                output.low_spk = low_spk;
                
                %CALCULATE PERFORMANCE
                Sgt = false(1,length(X)); Sgt(ST{cellN}) = true;
                Sm = false(1,length(X)); Sm(output.spikeTimes) = true;
                Smr = false(1,length(X)); Smr(output.rawROI.spikeTimes) = true;
                simOutput(simIter,sigSign).truePos(cellN) = sum(Sm & Sgt)./sum(Sgt);
                simOutput(simIter,sigSign).truePosRaw(cellN) = sum(Smr & Sgt)./sum(Sgt);
                
                simOutput(simIter,sigSign).falsePos(cellN) = sum(Sm & ~Sgt)./sum(Sm);
                simOutput(simIter,sigSign).falsePosRaw(cellN) = sum(Smr & ~Sgt)./sum(Smr);
                
                simOutput(simIter,sigSign).falseNeg(cellN) = sum(~Sm & Sgt)./sum(Sgt);
                simOutput(simIter,sigSign).falseNegRaw(cellN) = sum(~Smr & Sgt)./sum(Sgt);
                
                simOutput(simIter,sigSign).intersectOverUnion(cellN) = sum(Sm & Sgt)./sum(Sm | Sgt);
                simOutput(simIter,sigSign).intersectOverUnionRaw(cellN) = sum(Smr & Sgt)./sum(Smr | Sgt);
                
%                 selectSubThresh = ~conv(Sgt, ones(1,11),'same');
%                 simOutput(simIter).subThreshCorr(cellN) = corr(simOutput(simIter).gtHP(cellN,selectSubThresh)', output.y(selectSubThresh)');
%                 simOutput(simIter).subThreshCorrRaw(cellN) = corr(simOutput(simIter).gtHP(cellN,selectSubThresh)', output.rawROI.X(selectSubThresh)');
                
                simOutput(simIter,sigSign).est(cellN) = output;
                
                disp('Performance:')
                sum(Sm & Sgt)./sum(Sm | Sgt)
                
                if ~mod(cellN,5)
                    close all
                end
                drawnow;
            end
        close all
        end
        savefast([dr filesep fns{fn_ix}(1:end-4) '_SimResults_' nametag '.mat'], 'simOutput');
        toc
    end
end
end

function videoFilt = highpassVideo(video, freq, sampleRate)
normFreq = freq/(sampleRate/2);
[b,a] = butter(3,normFreq, 'high');
videoFilt = filtfilt(b,a,video);
end