function IntArrAnalysis_revised(infile,directory,probeName,ianalysis,numparticles,dateString,varargin)
%% Preamble
%  This script determines the inter-arrival time between peaks in a bimodal
%  distribution for each population of particles. Follows Field et al.
%  (2006) technique.
% 
%  Original code by Wei Wu, Univ. Illinois.
%  Significant modifications by Joe Finlon, Univ. Illinois 2017.
% 
%       **************************
%       *** Modification Notes ***
%       **************************
% 
% 
%  Usage:
%   infile : the particle-by-particle file containing inter-arrival times
%	directory: file path to save plots and threshold data
%	probeName : for handling SPEC probes (2DS & HVPS) differently
%	ianalysis: 1 to start inter-arrival analysis; 0 indicate only plotting
%   numparticles: population size belonging to each bimodal fit
%	dateString: yyyymmdd if not using parallel processing, otherwise
%       yyyymmdd_varargin format
%	varargin: {n chunks to be processed in parallel}{optional input for nth
%       chunk to be processed} -- length(varargin)==1 means parallel
%       processing is ignored
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import file variables

disp(['File: ', infile])

% Read the inter-arrival time data
ncid=netcdf.open(infile,'nowrite');
if strcmp(probeName,'2DS') || strcmp(probeName,'HVPS')
	tempTime=netcdf.getVar(ncid,netcdf.inqVarID(ncid,'Time_in_seconds'));
	intarr(1) = 0; % sets inter arrival time of first particle equal to 0
	intarr(2:length(tempTime)) = diff(tempTime); % subtract time between particles
	clear tempTime;
else
	intarr=netcdf.getVar(ncid,netcdf.inqVarID(ncid,'inter_arrival'));
end

% Read other particle variables
timehhmmss=netcdf.getVar(ncid,netcdf.inqVarID(ncid,'Time'));
date = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'Date'));
Dmax = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'image_diam_minR')); % Joe Finlon
netcdf.close(ncid)
intarr(intarr<=0)=NaN;
int_arr=intarr;

% Trim infile variable if processing will take too long
if length(varargin)==2 % two arguments, # CPUs and the nth chunk to process
    nChunks = varargin{1}; % number of chunks being processed in parallel
    fileNum = varargin{2}; % nth chunk to be processed
    startInd = 1000000*ceil(length(intarr)/(nChunks*1000000))*(fileNum-1)+1;
    endInd = min(1000000*ceil(length(intarr)/(nChunks*1000000))*fileNum,length(int_arr));
    
    timehhmmss = timehhmmss(startInd:endInd);
    date = date(startInd:endInd);
    int_arr = int_arr(startInd:endInd);
end


% % Plot Contour
% 
% bins = logspace(-7, 0, 70);
% binsCenter=bins(1:end-1)+diff(bins)/2;
% num_particles = 25000;                  % # of particles to factor into fit
% n=1;
% for i=1:num_particles:length(int_arr)    
%  	  indicies = i:min([i+num_particles-1 length(int_arr)]);
%   	  arr = int_arr(indicies);
%   	  [h,edges] = histcounts(arr, bins);
%       binsCenter=bins(1:end-1)+diff(bins)/2;
%       hist2dc(n,:) = h;
%       NEWtime2dc(n)=timehhmmss(i);
%       NEWdate2dc(n)=date(i);
%       n=n+1;
% end
% 
% 
% %%
% figure;
% figure('Visible','off');
% %scatter(hhmmss2insec(time2dc),intarr2dc,'.')
% hist2d=hist2dc;
% histsum=sum(hist2d,2);
% histsum=repmat(histsum,1,69);
% contourf(time2datenum(NEWdate2dc', NEWtime2dc'),binsCenter,(hist2dc./histsum)',0.005:0.005:0.1,'LineColor','none');
% colorbar;
% datetick('x','HH:MM');
% set(gca,'yscale','log')
% ylim([1e-7,1])
% xlabel('Time')
% ylabel('Inter-arrival Time [sec]')
% set(gca,'FontSize',20 )
% set(findall(gcf,'type','text'),'FontSize',20)
% savefig('2DSIntArrContour_20151201.fig')
% 
% % plot the interarrival time in dot scatter
% figure;
% figure('Visible','off');
% n=1;
% %time=hhmmss2insec(timehhmmss)/3600/24;
% time = time2datenum(date, timehhmmss);
% plot(time(1:n:end),intarr(1:n:end),'.','markersize', 0.5); 
% set(gca,'yscale','log');
% datetick('x','HH:MM')
% ylim([1e-7,1])
% xlabel('Time')
% ylabel('Inter-arrival Time [sec]')
% set(gca,'FontSize',20 )
% set(findall(gcf,'type','text'),'FontSize',20)
% savefig('2DSIntArrScatter_20151201.fig')

if (ianalysis==0)
    return
end

%% Start analyzing if prompted

clear hist2dc NEWdate2dc NEWtime2dc
n=1;
int_arr(int_arr<0)=0;
int_arr(int_arr>0.1)=0;
bins = logspace(-7, 0, 35);              % Specify range of normalized frequency histogram
width = log(bins(2))-log(bins(1));

% Determine # particles to factor into bimodal fit
if numel(find(timehhmmss==mode(timehhmmss)))<numparticles
    num_particles = numparticles; % mininum # of particles to factor into fit
else % # of particles to factor into fit to nearest 25k
    num_particles = (numparticles/4)*...
        ceil(numel(find(timehhmmss==mode(timehhmmss)))/(numparticles/4));
end
disp(['  ', num2str(num_particles), ' particles will be facored into the bimodal distribution fit.'])

% Initialize bimodal fitting function
%bimodalfit = @(tau, dt) (dt/tau(1)).*exp(-dt/tau(1)); % Equation from Chapter 2+3 for bimodal fit
bimodalfit = @(tau, dt) (1-tau(3)).*(dt/tau(1)).*exp(-dt/tau(1))+(tau(3)).*(dt/tau(2)).*exp(-dt/tau(2));
tau_std=[1e-2 1e-6 0.5];

hist2dc = NaN(ceil(length(int_arr)/num_particles), length(bins)-1);
NEWtime2dc = NaN(1, ceil(length(int_arr)/num_particles));
NEWdate2dc = NaN(1, ceil(length(int_arr)/num_particles));

% Loop through sample populations and determine threshold
for i=1:num_particles:length(int_arr)
      disp(['  Current Progress: ', num2str(i), '/', num2str(length(int_arr)),...
          ' Time: ', datestr(now)])
      
 	  indicies = i:min([i+num_particles-1 length(int_arr)]);
  	  arr = int_arr(indicies);
  	  [h,~] = histcounts(arr, bins);
      binsCenter = bins(1:end-1)+diff(bins)/2;
      hist2dc(n,:) = h;
      NEWtime2dc(n) = timehhmmss(i);
      NEWdate2dc(n) = date(i);

      beta0 = [1e-2 1e-6 0.5]; % initial parameter guesses
      [tau_std] = abs(nlinfit(binsCenter,h./sum(h)./width,bimodalfit,beta0,...
          statset('Robust', 'on', 'FunValCheck', 'off', 'MaxIter', 1000)));
      
      if sum(isnan(tau_std))>0 % bimodal fit still isn't achieved (nlinfit returns NaN) -- added by Joe Finlon
          threshhold(i:min([i+num_particles-1 length(int_arr)])) = 1.3494e-6; % use a default inter-arrival time
      else
          threshhold(i:min([i+num_particles-1 length(int_arr)]))=min(tau_std(1:2))*2;
      end
            
      for j=i:min([i+num_particles-1 length(int_arr)])
        tau_all(j,:)=tau_std;
      end
      
      dt = binsCenter;
      tau = tau_std;
      hfit = (1-tau(3)).*(dt/tau(1)).*exp(-dt/tau(1))+(tau(3)).*(dt/tau(2)).*exp(-dt/tau(2));
      
      % Determine thresholds using 3 different techniques
      tau1 = max(tau_std(1:2)); tau2 = min(tau_std(1:2));
      newDT = dt(dt<tau1 & dt >tau2);
      
      if isempty(newDT) %% JOE FINLON
          newDT = dt([find(dt<tau1 , 1, 'last'), find(dt>tau2, 1, 'first')]);
          [minFit, indexFit] = min( hfit([find(dt<tau1 , 1, 'last'), find(dt>tau2, 1, 'first')]) );
          [minOriginal, indexOriginal] = min( h([find(dt<tau1 , 1, 'last'), find(dt>tau2, 1, 'first')]) );
      else
          [minFit, indexFit] = min( hfit(dt <tau1 & dt >tau2 ) );
          [minOriginal, indexOriginal] = min( h(dt<tau1 & dt >tau2 ) );
      end
      
      if isempty(newDT) % bimodal fit isn't achieved (nlinfit returns NaN) -- added by Joe Finlon
          disp(['Trouble obtaining a bimodal fit for index ', num2str(i),...
              '. Setting inter-arrival thresholds to a default value.'])
          threshhold_ww(i:min([i+num_particles-1 length(int_arr)])) = 1.3494e-6; % use a default inter-arrival time
          threshhold_ak(i:min([i+num_particles-1 length(int_arr)])) = 1.3494e-6; % use a default inter-arrival time
      else
          threshhold_ww(i:min([i+num_particles-1 length(int_arr)]))=newDT(indexFit);
          threshhold_ak(i:min([i+num_particles-1 length(int_arr)]))=newDT(indexOriginal);
      end
      
      % Optionally plot inter-arrival information for current population
      if sum(isnan(tau_std))==0
        figure('visible','off'); set(gcf, 'color', 'w');
        bar(bins(1:end-1), h ./ sum(h) ./ width, 'histc'); hold on;
        plot(dt, hfit, 'k');
        plot(ones(1,length(0:0.01:0.3))*(tau2),0:0.01:0.3,'--k'); % tau2
        plot(ones(1,length(0:0.01:0.3))*(tau1),0:0.01:0.3,'-.k'); % tau1
        plot(ones(1,length(0:0.01:0.3))*(min(tau_std(1:2))*2),0:0.01:0.3,'g'); % threshold
        plot(ones(1,length(0:0.01:0.3))*(newDT(indexFit)),0:0.01:0.3,'b'); % threshold_ww
        plot(ones(1,length(0:0.01:0.3))*(newDT(indexOriginal)),0:0.01:0.3,'r'); % threshold_ak
      
        ylim([0 0.5]); set(gca, 'xscale', 'log'); set(gca, 'xminortick', 'on');
        %set(gca,'FontSize',16); set(findall(gcf,'type','text'),'FontSize',16);
        title(['Inter-arrival Distribution (', num2str(NEWtime2dc(n)), ' UTC)']);
        xlabel('Inter-arrival Time (s)'); ylabel('Frequency');
        legend({'frequency', 'bin endpoint', 'fit', 'lower peak', 'higher peak',...
            '2*(lower peak)', 'freq. min between peaks from fit', 'freq. min between peaks in hist'},...
            'FontSize', 6, 'Location', 'northwest');
        print([directory, probeName, 'IntArrHistogram_', num2str(NEWtime2dc(n)), '.',...
            dateString, '.jpg'],'-djpeg','-r300')
      end
      
      n=n+1;
end

%% Plotting Routines

bins = logspace(-7, 0, 70);
binsCenter = bins(1:end-1)+diff(bins)/2;

disp(['  ', num2str(num_particles), ' paricles are factored into the contour plot for each time interval.'])
n=1;
for i=1:num_particles:length(int_arr)    
 	  indicies = i:min([i+num_particles-1 length(int_arr)]);
  	  arr = int_arr(indicies);
  	  [h,~] = histcounts(arr, bins);
      binsCenter = bins(1:end-1)+diff(bins)/2;
      hist2dc_contour(n,:) = h;
      NEWtime2dc(n) = timehhmmss(i);
      NEWdate2dc(n) = date(i);
      n=n+1;
end
NEWtime2dc(n-1) = timehhmmss(length(int_arr)); % fix end time for contour plot
NEWdate2dc(n-1) = date(length(int_arr)); % fix end date for contour plot

histsum = sum(hist2dc_contour,2);
histsum = repmat(histsum,1,69);
time = time2datenum(date, timehhmmss);

% % ======= Plot the inter-arrival time in dot scatter =======
% figure('visible','off'); set(gcf, 'color', 'w');
% n=1;
% plot(time(1:n:end),intarr(1:n:end),'.','markersize', 0.5);

% ======= Plot the inter-arrival time as a contoured distribution =======
figure('visible','off'); set(gcf, 'color', 'w');
contourf(time2datenum(NEWdate2dc', NEWtime2dc'),binsCenter,(hist2dc_contour./histsum)',...
    0.005:0.005:0.1,'LineColor','none');
colormap(jet); colorbar; hold on;
n=1;
plot(time(1:n:length(time)),threshhold(1:n:end),'g','LineWidth',2);
plot(time(1:n:length(time)),threshhold_ww(1:n:end),'b','LineWidth',2);
plot(time(1:n:length(time)),threshhold_ak(1:n:end),'r','LineWidth',2);

ylim([1e-7, 1]); set(gca,'yscale','log'); datetick('x','HH:MM');
title(sprintf('Inter-arrival Time Frequency for %s',dateString));
xlabel('Time'); ylabel('Inter-arrival Time [sec]');
legend({'frequency', '2*lower peak', 'freq. min between peaks from fit',...
    'freq. min between peaks in hist'}, 'FontSize', 6, 'Location', 'northwest');
% set(gca,'FontSize',16); set(findall(gcf,'type','text'),'FontSize',16);


% savefig([directory, probeName, 'IntArrAnalysis.', dateString, '.fig'])
print([directory, probeName, 'IntArrAnalysis.', dateString, '.jpg'],'-djpeg','-r300')

% ======= Box plot of inter-arrival times by size =======
dD = 0.1; % bin width for size categories to partition inter-arrival times [mm]
binsMid = 0.15:dD:1.25; % particle size categories to partition inter-arrival times [mm]

intArr_copy = int_arr; % copy inter-arrival times for plotting
intArr_sizes = NaN(1, length(int_arr)); % allocate size category for 
for iter=1:length(binsMid)
    intArr_sizes(find((Dmax>=binsMid(iter)-dD/2) & (Dmax<binsMid(iter)+dD/2))) = binsMid(iter);
end

intArr_copy(find(isnan(intArr_sizes))) = []; % ignore particles outside of size range
intArr_sizes(find(isnan(intArr_sizes))) = []; % ignore particles outside of size range

figure('visible','off'); set(gcf, 'color', 'w');
intArr_round = NaN(1,ceil(length(int_arr)/num_particles)); % dummy inter-arrival for box plot grouping
for i=1:num_particles:length(int_arr)
    intArr_round(i) = 1000*(ceil(Dmax(i)*10)/10 - 0.05); % round to mid-point of 100 um bin incriments [e.g. 50,150,250]
end

boxplot(intArr_copy, 1000*intArr_sizes, 'PlotStyle',  'compact');
ylim([1e-7 1]); set(gca,'YScale','log');
title(sprintf('Inter-arrival Time by Size for %s',dateString));
xlabel('Size Bin Midpoint (\mum)'); ylabel('Inter-arrival Time [sec]');

% savefig([directory, probeName, 'IntArrSizes.', dateString, '.fig'])
print([directory, probeName, 'IntArrSizes.', dateString, '.jpg'],'-djpeg','-r300')

%% Save the Data

fprintf('Now writing ouput to file: %s\n\n', datestr(now));

% Save all variables to Matlab datafile
save([directory, probeName, 'IntArrAnalysis.', dateString, '.mat'])

% Save only threshold information for sizeDist.m script (may need manual
% intervention if fitting technique is not robust
f = netcdf.create([directory, 'intArrThreshold_', dateString, '.cdf'], 'clobber');

dimid0 = netcdf.defDim(f,'particleTime',length(time));

NC_GLOBAL = netcdf.getConstant('NC_GLOBAL');
netcdf.putAtt(f, NC_GLOBAL, 'Software', 'UIOPS/IntArrAnalysis_revised');
netcdf.putAtt(f, NC_GLOBAL, 'Institution', 'Univ. Illinois, Dept. Atmos. Sciences');
netcdf.putAtt(f, NC_GLOBAL, 'Creation Time', datestr(now, 'yyyy/mm/dd HH:MM:SS'));
netcdf.putAtt(f, NC_GLOBAL, 'Description', ['Contains inter-arrival threshold ',...
    'information for each particle following Field et al. (2006)']);
netcdf.putAtt(f, NC_GLOBAL, 'Flight Date', dateString)
netcdf.putAtt(f, NC_GLOBAL, 'Data Source', infile);
netcdf.putAtt(f, NC_GLOBAL, 'Probe Type', probeName);
netcdf.putAtt(f, NC_GLOBAL, 'Population Size', [num2str(num_particles),...
    ' particles per distribution fit']);

varid0 = netcdf.defVar(f,'particle_time','double',dimid0); 
netcdf.putAtt(f, varid0,'units','HHMMSS');
netcdf.putAtt(f, varid0,'name','Time');

varid1 = netcdf.defVar(f,'threshold','double',dimid0); 
netcdf.putAtt(f, varid1,'units','sec');
netcdf.putAtt(f, varid1,'name','Inter-arrival time threshold in n-particle blocks');

netcdf.endDef(f)

netcdf.putVar ( f, varid0, timehhmmss );
netcdf.putVar ( f, varid1, threshhold_ak );

netcdf.close(f) % Close output NETCDF file

disp('Finished determining inter-arrival time thresholds!')
end

function [dateValue] = time2datenum(date, timehhmmss)

secFromMidnight = floor(double(timehhmmss)/10000)*3600 + floor(mod(double(timehhmmss),10000)/100)*60 +...
    floor(mod(double(timehhmmss),100));
dateVector = [floor(double(date)/10000), floor(mod(double(date),10000)/100), floor(mod(double(date),100)) ,...
    zeros(length(date),1), zeros(length(date),1), secFromMidnight];
dateValue = datenum(dateVector);

end