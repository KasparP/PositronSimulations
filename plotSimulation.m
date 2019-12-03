function out = plotSimulation (simOutput)

metrics = {'truePos', 'falsePos', 'intersectOverUnion'};
params = {'spikeAmp', 'nCells'};


for p_ix = 1:length(params)
    Ps = [simOutput(:,1).(params{p_ix})];
    Pvals = unique(Ps); Pvals(Pvals==5) = [];
    if length(Pvals)>1 % if this simulation changed this parameter
        Mperf = []; Eperf = [];
        for m = 1:length(metrics)
            for NCix = length(Pvals):-1:1
                P = Pvals(NCix); perf = [];
                
                sel =  Ps==P & (cellfun(@length, {simOutput(:,1).(metrics{m})})== cellfun(@length, {simOutput(:,2).(metrics{m})}));
                if ~any(sel)
                    continue
                end
                perf(:,1) = [simOutput(sel,1).(metrics{m})];
                perf(:,2) = [simOutput(sel,2).(metrics{m})];
                
                Mperf(NCix,:) = mean(perf,1);
                Eperf(NCix,:) = std(perf,0,1)./sqrt(size(perf,1));
            end
            figure('Name', metrics{m}),
            errorbar(Pvals, 1-Mperf(:,1), Eperf(:,1), 'r')
            hold on
            errorbar(Pvals, 1-Mperf(:,2), Eperf(:,2), 'b')
            plot(Pvals, 1-Mperf(:,1), 'ro')
            plot(Pvals, 1-Mperf(:,2), 'bo')
            legend({'Negative-going', 'Positive-going'})
            xlabel(params{p_ix}); ylabel(metrics{m})
            set(gca, 'yscale', 'log')
            
            if m==3
                out.param = params{p_ix};
                out.ydata(:,:) = 1-Mperf;
                out.xdata = Pvals;
                out.edata(:,:) = Eperf;
            end
        end
    end
end
end