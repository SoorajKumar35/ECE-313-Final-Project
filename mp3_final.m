%% Task 0
% loading the patient data file into MATLAB for all 9 patients into readable
% variables

commandwindow;

pat1 = load('1_a41178.mat');
pat2 = load('2_a42126.mat');
pat3 = load('3_a40076.mat');
pat4 = load('4_a40050.mat');
pat5 = load('5_a41287.mat');
pat6 = load('6_a41846.mat');
pat7 = load('7_a41846.mat');
pat8 = load('8_a42008.mat');
pat9 = load('9_a41846.mat');

% parsing them into array of patient data

pat_array = [pat1, pat2, pat3, pat4, pat5, pat6, pat7, pat8, pat9];

% opening the result file

fid = fopen('ECE313_FinalProject_Group12','w');

% for each of the data files, apply 'floor' on the data set in order to
% convert from double to integer

pat_array(1).all_data = floor(pat_array(1).all_data);
pat_array(2).all_data = floor(pat_array(2).all_data);
pat_array(3).all_data = floor(pat_array(3).all_data);
pat_array(4).all_data = floor(pat_array(4).all_data);
pat_array(5).all_data = floor(pat_array(5).all_data);
pat_array(6).all_data = floor(pat_array(6).all_data);
pat_array(7).all_data = floor(pat_array(7).all_data);
pat_array(8).all_data = floor(pat_array(8).all_data);
pat_array(9).all_data = floor(pat_array(9).all_data);

% for each of the data files store number of data points for training
% by creating 1D array of 9 elements and populating it, since
% each patient data file has different number of samples, we need to repeat
% this 9 times

training_data_points = zeros(1, 9);
testing_data_points = zeros(1, 9);

for i = 1:9
    len = length(pat_array(i).all_data);
    training_data_points(i) = floor((2/3)*(len));                % added floor to eradicate error
    testing_data_points(i) = (len) - (training_data_points(i));
end

% initialize the training, testing, label_training and label_testing
% datasets now for each patient data set

for i = 1:9
    training(i).all_data = pat_array(i).all_data(:,1:training_data_points(i));
    training(i).all_labels = pat_array(i).all_labels(:,1:training_data_points(i));
end

for i = 1:9
    lenn = length(pat_array(i).all_data);
    testing(i).all_data = pat_array(i).all_data(:,training_data_points(i)+1:(lenn));
    testing(i).all_labels = pat_array(i).all_labels(:,training_data_points(i)+1:(lenn));
end

%% Task 1.1
%% 1.1 (a)
% The following attempts to calculate prior probabilities P(H0) and P(H1)
% by summing up all the training labels and dividing it by the number
% training data points

% for each patient data set the probability is calculated

% Prior probability array for all 9 data sets is initialized below and then
% populated

P_H1 = zeros(1, 9);
P_H0 = zeros(1, 9);
total = zeros(1, 9);

for i = 1:9
    total(i) = sum(training(i).all_labels);
end

for i = 1:9
    P_H1(i) = total(i)/(training_data_points(i));
    P_H0(i) = 1 - P_H1(i);
end

%% 1.1 (b)

% in order to help construct the likelihood l_matrix we first
% attempt to make data for golden alarms (H1) and non-golden alarms (H0)
% this can be done by looping through the training labels and checking for
% where '1's and '0's occur, this is done for each patient data set

for i = 1:9

    j = 1;

    while (j < 8)

        golden = 1;                                                                           % keep track of indices
        non_golden = 1;

        for k = 1:training_data_points(i)

            if (training(i).all_labels(1, k))
                training(i).golden_data(j, golden) = training(i).all_data(j, k);
                golden = golden + 1;                                                          % increment index

            else
                training(i).non_golden_data(j, non_golden) = training(i).all_data(j, k);
                non_golden = non_golden + 1;
            end

        end

     j = j+1;

    end

end

% The following code attempts to construct likelihood matrices for each of
% the 7 columns in the all_data set for each of the 9 data sets

l_matrix = cell(9, 7);
length_array = zeros(9, 7);

% This returns a 9 X 7 cell array of empty matrices
% The following code attempts to populate this cell
% by looping over every row and column and filling in
% PMF values based on golden or non golden data

for i = 1:9

    for j = 1:7

        golden = training(i).golden_data(j,:);

        %gold_tab = tabulate(golden)';                                        % determine frequency table for golden data
        gold_tab = tabulate(floor(golden))';
        nongolden = training(i).non_golden_data(j,:);
        %nongolden_tab = tabulate(nongolden)';                                % determine frequency table for nongolden data
        nongolden_tab = tabulate(floor(nongolden))';

        %combined_tab = union(gold_tab, nongolden_tab);                          % set union both the frequency tables
        min_gold_tab = min(gold_tab(1,:));
        max_gold_tab = max(gold_tab(1,:));
        min_nongolden_tab = min(nongolden_tab(1,:));
        max_nongolden_tab = max(nongolden_tab(1,:));


        fprintf('value of nongoldtab_min %d, mingoldtab %d\n', min_nongolden_tab, min_gold_tab);
        lesser_range = min(min_gold_tab, min_nongolden_tab);
        max_range = max(max_gold_tab,max_nongolden_tab);
        combined_tab =  zeros(3,max_range-lesser_range+1);
        fprintf('value of lesser range %d and value of displacement %d\n', lesser_range,0 - lesser_range + 1);
        for k = lesser_range:max_range
            indexnong = find(nongolden_tab(1,:)==k);
            indexg = find(gold_tab(1,:)==k);
            displacement = 0 - lesser_range + 1;
            combined_tab(1,k+displacement) = k;
            if isempty(indexg)
                combined_tab(2,k+displacement) = 0;
            else
                combined_tab(2,k+displacement) = (gold_tab(3,indexg))/(100);
            end

            if isempty(indexnong)
                combined_tab(3,k+displacement) = 0;
            else
                combined_tab(3,k+displacement) = (nongolden_tab(3,indexnong))/(100);
            end
%ML
        if(combined_tab(3, k+displacement) > combined_tab(2,k+displacement))
            combined_tab(4,k+displacement) = 0;
        else
            combined_tab(4,k+displacement) = 1;
        end
%MAP
        if(P_H0(i)*combined_tab(3, k+displacement) > P_H1(i)*combined_tab(2, k+displacement))
            combined_tab(5, k+displacement) = 0;
        else
            combined_tab(5,k+displacement) = 1;
        end

        end

    abc{i,j} = combined_tab';


%         final_length = length(combined_tab);                                    % determine the length to loop over in order to populate
%
%         length_array(i, j) = final_length;
%
%         k = 1;
%
%         while(k < final_length+1)
%
%             l_matrix{i, j}(1, k) = combined_tab(k);                             % begin to populate the l_matrix
%
%             bool1 = ismembertol(combined_tab(k), gold_tab(1,:));                % check if current index exists in both
%             bool2 = ismembertol(combined_tab(k), nongolden_tab(1,:));
%
%             if (bool1 == true)
%                 index1 = find(gold_tab(1,:) == combined_tab(k), 1);             % and look up column 3 in the tab freq table for %
%                 temp1 = gold_tab(3, index1);
%                 value1 = (temp1) / (100);                                   % divide by 100, to make value b/w 0 and 1
%                                                                             % if it does, determine index using find in combined_tab
%                 l_matrix{i, j}(2, k) = value1;                              % if it doesn't, populate h1 row's feature to 0
%             else
%                 l_matrix{i, j}(2, k) = 0;
%             end
%
%             if (bool2 == true)
%                 index2 = find(nongolden_tab(1,:) == combined_tab(k), 1);
%                 temp2 = nongolden_tab(3, index2);
%                 value2 = (temp2) / (100);
%                 l_matrix{i, j}(3, k) = value2;
%             else
%                 l_matrix{i, j}(3, k) = 0;
%             end
%
%             k = k+1;
%
%         end




    end
end

%% 1.1 (c)

% The following code generates 9 figures, one for each patient which contain
% 7 subplots for each of the 7 features, in each subplotthe pmf is plotted
% under both golden and non_golden data

labels = {'Mean Area under the heartbeat','Mean R-to-R peak interval','Number of beats per minute (Heart Rate)','Peak to peak interval for blood pressure','Systolic blood pressure','Diastolic blood pressure','Pulse pressure'};
max_length = [23, 226, 228, 225, 106, 78, 48];

% max_length determined by printing out l_matrix

%for i = 1:9
%    for j = 1:7
%    fprintf("%d ", length_array(i, j));
%    end
%    fprintf("\n");
%end

for i = 1:9

   figure;                                                                  % create a figure

    for j = 1:7

        subplot(7, 1, j);                                                   % create a subplot

        tempo = abc{i, j}(:,3);
        maxtempo = max(tempo);
        variable = abc{i, j}(:,1);

        tempo1 = labels(j);
        tempo2 = max_length(j);

        tempo3 = abc{i, j}(:,2);
        maxtempo3 = max(tempo3);

        p = plot(variable, tempo, '-b');                                              % populate it
        p.LineWidth = 0.95;

        title(tempo1);

        axis([0 tempo2 0 max(maxtempo, maxtempo3)]);                                                % set x and y max/min values

        hold on;

        q =  plot(variable, tempo3, '-r');
        q.LineWidth = 0.95;

    end

    lgd = legend('H0 Probability Mass Function', 'H1 Probability Mass Function', 'Location', 'bestoutside');
    lgd.TextColor = 'black';
    title(lgd, 'PMFs');

end

%% 1.1 (d)

% The following code attempts to calculate MP and MAP decision rule vectors
% by breaking ties according to the final project ppt

% we loop over every value in the l_matrix checking with prior probabilities
% too

% for i = 1:9
%
%     for j = 1:7
%
%         val = l_matrix{i, j}(1,:);                                              % get the length of the first l_matrix for the jth feature
%         length_val = length(val);
%
%         for k = 1:length_val
%
%             x = l_matrix{i, j}(2,k);
%             y = l_matrix{i, j}(3,k);                                            % get variables to compare and set from the l_matrix
%
%             if (x >= y)                                                         % if H1_pmf >= H0_pmf then ML vector = 1
%                 l_matrix{i, j}(4, k) = 1;
%             else
%                 l_matrix{i, j}(4, k) = 0;
%                                                                                 % else, ML vector = 0
%             end
%
%             xx = (x)*(P_H1(i));                                                 % using prior probabilities here
%             yy = (y)*(P_H0(i));
%
%             if (xx >= yy)                                                       % same for MAP vector
%                 l_matrix{i, j}(5, k) = 1;
%             else
%                 l_matrix{i, j}(5, k) = 0;
%             end
%         end
%     end
% end

% The ML and MAP vectors occupy the 4th and 5th l_matrix in the cell array
% respectively

%% 1.1 (e)

% The following code saves the results of everything done above in the form
% of a 9 by 7 call array, called HT_table_array with each vell representing
% a two-dimensional array

% HT_table_array = cell(9, 7);
%
% % populate this array by copying values over from the l_matrix created above
%
% for i = 1:9
%     for j = 1:7
%         HT_table_array{i, j} = l_matrix{i, j};
%     end
% end
HT_table_array = abc;

%% Task 1.2
dr_alarms_array = cell(9,7);

for j = 1:7
    for i = 1:9
        dr_alarms = zeros(2, size(testing(i).all_data,2));
        dr_alarms_array(i,j) = mat2cell(dr_alarms, 2, size(testing(i).all_data ,2));
    end
end

Error_table_array = cell(9,7);

for z = 1:9
    for j = 1:7
        for i = 1:size(testing(z).all_data,2)
            HT_table_req = HT_table_array{z,j};
            test = testing(z).all_data(j,i);
            [row,col] = find(HT_table_req == test);
            if ((size(row,1) == 0) || (size(col,1) == 0))
                message_1 = 'Could not find index';
                dr_alarms_array{z,j}(1,i) = 1;
                dr_alarms_array{z,j}(2,i) = 1;
            else
%             fprintf('The value of i = %d\n', i);
%             fprintf('The size of col = %d\n', col(1,1));
%             fprintf('The dim of HT_table_array %d x %d\n', size(HT_table_req,1), size(HT_table_req,2));

            dr_alarms_array{z,j}(1,i) = HT_table_req(4,col(1,1));
            dr_alarms_array{z,j}(2,i) = HT_table_req(5,col(1,1));
            end
        end
    end
end

for z = 1:7
    for j = 1:9
        false_alarm_ml = 0;
        miss_detection_ml = 0;
        false_alarm_map = 0;
        miss_detection_map = 0;
        for i = 1:size(testing(j).all_data,2)
            if dr_alarms_array{j,z}(1,i) == 1 && testing(j).all_labels(i) == 0
                false_alarm_ml = false_alarm_ml + 1;
            elseif dr_alarms_array{j,z}(1,i) == 0 && testing(j).all_labels(i) == 1
                miss_detection_ml = miss_detection_ml + 1;
            end
            if dr_alarms_array{j,z}(2,i) == 1 && testing(j).all_labels(i) == 0
                false_alarm_map = false_alarm_map + 1;
            elseif dr_alarms_array{j,z}(2,i) == 0 && testing(j).all_labels(i) == 1
                miss_detection_map = miss_detection_map + 1;
            end
        end
        Error_table = zeros(2,3);
        %Error_table(1,1) = (sum(dr_alarms_array{z,j}(1,:)) + (size(testing(z).all_labels,2)-sum(testing(z).all_labels) )) / (size(dr_alarms_array{z,j},2) + size(testing(z).all_labels,2));
        Error_table(1,1) = false_alarm_ml/size(testing(j).all_labels,2);
        num1 = Error_table(1,1);
        Error_table(1,1) = Error_table(1,1)/((size(testing(j).all_labels,2) - sum(testing(j).all_labels))/size(testing(j).all_labels,2));
        %Error_table(1,2) = (sum(testing(z).all_labels) + (size(dr_alarms_array{z,j},2) - sum(dr_alarms_array{z,j}(1,:)))) / ((size(dr_alarms_array{z,j},2) + size(testing(z).all_labels,2)));
        Error_table(1,2) = miss_detection_ml/size(testing(j).all_labels,2);
        num2 = Error_table(1,2);
        Error_table(1,2) = Error_table(1,2)/(sum(testing(j).all_labels)/size(testing(j).all_labels,2));
        first_arg = (P_H0(j)*num1);
        second_arg = (P_H1(j)*num2);
        Error_table(1,3) = first_arg + second_arg;

        Error_table(2,1) = false_alarm_map/size(testing(j).all_labels,2);
        Error_table(2,2) = miss_detection_map/size(testing(j).all_labels,2);
        num3 = Error_table(2,1);
        num4 = Error_table(2,2);
        Error_table(2,1) = Error_table(2,1)/((size(testing(j).all_labels,2) - sum(testing(j).all_labels))/size(testing(j).all_labels,2));
        Error_table(2,2) = Error_table(2,2)/(sum(testing(j).all_labels))/size(testing(j).all_labels,2);
        first_arg_map = (P_H0(j)*num3);
        second_arg_map = (P_H1(j)*num4);
        Error_table(2,3) = first_arg_map + second_arg_map;

        Error_table_array{j,z} = mat2cell(Error_table,2,3);
    end
end

%% Task 2.1(a)
%It is known that there is a problem in one of the nine patient data.
%Note that, ideally, the data for each patient would be distinct. For
%the accuracy of the analysis, you need to identify the problematic data.

%Using the corrcoef function in MATLAB, analyze the correlation
%since we see that all the data sets for each patient are of different
%lenghts, we must try and take the shortest length of all the data sets.
numpat1data = size(pat1.all_data, 2);
numpat2data = size(pat2.all_data, 2);
numpat3data = size(pat3.all_data, 2);
numpat4data = size(pat4.all_data, 2);
numpat5data = size(pat5.all_data, 2);
numpat6data = size(pat6.all_data, 2);
numpat7data = size(pat7.all_data, 2);
numpat8data = size(pat8.all_data, 2);
numpat9data = size(pat9.all_data, 2);

numpatarray = [numpat1data, numpat2data, numpat3data, numpat4data, numpat5data, numpat6data, numpat7data, numpat8data, numpat9data];
lowestnum = min(numpatarray);
nine = 9;
seven = size(pat1.all_data, 1);
numleng = find(numpatarray == lowestnum);
val = training_data_points(numleng);

% % training(1).short(1,1412) = training(1).all_data(1:1412);
% % training(2) = training(2).all_data(1:val);
% % training(3) = training(3).all_data(1:val);
% % training(4) = training(4).all_data(1:val);
% % training(5) = training(5).all_data(1:val);
% % training(6) = training(6).all_data(1:val);
% % training(7) = training(7).all_data(1:val);
% % training(8) = training(8).all_data(1:val);
% % training(9) = training(9).all_data(1:val);
%
% % We must make the datasets smaller so that they all align and populate the
% % alter column
% for k = 1:nine
%         for h = 1:val
%         training(k).alter(1, h) = training(k).all_data(1,h);
%         end
%         for h = 1:val
%         training(k).alter(2, h) = training(k).all_data(2,h);
%         end
%         for h = 1:val
%         training(k).alter(3, h) = training(k).all_data(3,h);
%         end
%         for h = 1:val
%         training(k).alter(4, h) = training(k).all_data(4,h);
%         end
%         for h = 1:val
%         training(k).alter(5, h) = training(k).all_data(5,h);
%         end
%         for h = 1:val
%         training(k).alter(6, h) = training(k).all_data(6,h);
%         end
%         for h = 1:val
%         training(k).alter(7, h) = training(k).all_data(7,h);
%         end
% end
% %calcule the correlation coefficients
% for k = 1:nine
%     for j = 1:seven
%             correlation = corrcoef(training(k).alter(j,:), training(k).alter(1,:));
%             training(k).correchart(j,1) = correlation(2,1);
%             correlation = corrcoef(training(k).alter(j,:), training(k).alter(2,:));
%             training(k).correchart(j,2) = correlation(2,1);
%             correlation = corrcoef(training(k).alter(j,:), training(k).alter(3,:));
%             training(k).correchart(j,3) = correlation(2,1);
%             correlation = corrcoef(training(k).alter(j,:), training(k).alter(4,:));
%             training(k).correchart(j,4) = correlation(2,1);
%             correlation = corrcoef(training(k).alter(j,:), training(k).alter(5,:));
%             training(k).correchart(j,5) = correlation(2,1);
%             correlation = corrcoef(training(k).alter(j,:), training(k).alter(6,:));
%             training(k).correchart(j,6) = correlation(2,1);
%             correlation = corrcoef(training(k).alter(j,:), training(k).alter(7,:));
%             training(k).correchart(j,7) = correlation(2,1);
%     end
% end
%
% %find the lowest of the coefficients for each patient (9)
%
% %Patient 1
%     for j = 1:7
%         relation = min(abs(training(1).correchart(:)));
%     end
%     training(1).least_correlated = relation;
% %Patient 2
%     for j = 1:7
%         relation = min(abs(training(2).correchart(:)));
%     end
%     training(2).least_correlated = relation;
% %Patient 3
%     for j = 1:7
%         relation = min(abs(training(3).correchart(:)));
%     end
%     training(3).least_correlated = relation;
% %Patient 4
%     for j = 1:7
%         relation = min(abs(training(4).correchart(:)));
%     end
%     training(4).least_correlated = relation;
% %Patient 5
%     for j = 1:7
%         relation = min(abs(training(5).correchart(:)));
%     end
%     training(5).least_correlated = relation;
% %Patient 6
%     for j = 1:7
%         relation = min(abs(training(6).correchart(:)));
%     end
%     training(6).least_correlated = relation;
% %Patient 7
%     for j = 1:7
%         relation = min(abs(training(7).correchart(:)));
%     end
%     training(7).least_correlated = relation;
% %Patient 8
%     for j = 1:7
%         relation = min(abs(training(8).correchart(:)));
%     end
%     training(8).least_correlated = relation;
% %Patient 9
%     for j = 1:7
%         relation = min(abs(training(9).correchart(:)));
%     end
%     training(9).least_correlated = relation;
%
% %least correlated features
% for k = 1:nine
%     for j = 1:seven
%         for h = 1:seven
%             if abs(training(k).correchart(j,h)) == training(k).least_correlated(1)
%                 training(k).corr_feat(1) = j;
%                 training(k).corr_feat(2) = h;
%             end
%         end
%     end
% end
for i = 1:seven
    for j = 1:4222 %lesser data points
        pat1.fit(i,j) = pat1.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat1.fit(i, :), pat2.all_data(i,:));
    corr.onetwo(i) = correlation(2,1);
end
%onethree
for i = 1:seven
    for j = 1:4296 %lesser data points
        pat1.fit(i,j) = pat1.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat1.fit(i, :), pat3.all_data(i,:));
    corr.onethree(i) = correlation(2,1);
end
%onefour
for i = 1:seven
    for j = 1:3010 %lesser data points
        pat1.fita(i,j) = pat1.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat1.fita(i, :), pat4.all_data(i,:));
    corr.onefour(i) = correlation(2,1);
end

%onefive
for i = 1:seven
    for j = 1:4299 %lesser data points
        pat5.fit(i,j) = pat5.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat5.fit(i, :), pat1.all_data(i,:));
    corr.onefive(i) = correlation(2,1);
end

%onesix
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat1.fitb(i,j) = pat1.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat1.fitb(i, :), pat6.all_data(i,:));
    corr.onesix(i) = correlation(2,1);
end
%oneseven
for i = 1:seven
    for j = 1:4298 %lesser data points
        pat1.fitb(i,j) = pat1.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat1.fitb(i, :), pat7.all_data(i,:));
    corr.oneseven(i) = correlation(2,1);
end

%oneeight
for i = 1:seven
    for j = 1:2118 %lesser data points
        pat1.fitc(i,j) = pat1.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat1.fitc(i, :), pat8.all_data(i,:));
    corr.oneeight(i) = correlation(2,1);
end

%onening
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat1.fitc(i,j) = pat1.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat1.fitc(i, :), pat9.all_data(i,:));
    corr.onenine(i) = correlation(2,1);
end


%twothree
for i = 1:seven
    for j = 1:4222 %lesser data points
        pat3.fitc(i,j) = pat3.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat3.fitc(i, :), pat2.all_data(i,:));
    corr.twothree(i) = correlation(2,1);
end
%twofour
for i = 1:seven
    for j = 1:3010 %lesser data points
        pat2.fitc(i,j) = pat2.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat2.fitc(i, :), pat4.all_data(i,:));
    corr.twofour(i) = correlation(2,1);
end
%twofive
for i = 1:seven
    for j = 1:4222 %lesser data points
        pat5.fitc(i,j) = pat5.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat5.fitc(i, :), pat2.all_data(i,:));
    corr.twofive(i) = correlation(2,1);
end

%twosix
for i = 1:seven
    for j = 1:4222 %lesser data points
        pat6.fitc(i,j) = pat6.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat6.fitc(i, :), pat2.all_data(i,:));
    corr.twosix(i) = correlation(2,1);
end
%twoseven
for i = 1:seven
    for j = 1:4222 %lesser data points
        pat7.fitc(i,j) = pat7.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat7.fitc(i, :), pat2.all_data(i,:));
    corr.twoseven(i) = correlation(2,1);
end
%twoeight
for i = 1:seven
    for j = 1:2118 %lesser data points
        pat2.fitd(i,j) = pat2.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat2.fitd(i, :), pat8.all_data(i,:));
    corr.twoeight(i) = correlation(2,1);
end
%twonine
for i = 1:seven
    for j = 1:4222 %lesser data points
        pat9.fitd(i,j) = pat9.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat9.fitd(i, :), pat2.all_data(i,:));
    corr.twonine(i) = correlation(2,1);
end
%threefour
for i = 1:seven
    for j = 1:3010 %lesser data points
        pat3.fitd(i,j) = pat3.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat3.fitd(i, :), pat4.all_data(i,:));
    corr.threefour(i) = correlation(2,1);
end
%threefive
for i = 1:seven
    for j = 1:4296 %lesser data points
        pat5.fitd(i,j) = pat5.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat5.fitd(i, :), pat3.all_data(i,:));
    corr.threefive(i) = correlation(2,1);
end
%threesix
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat3.fitd(i,j) = pat3.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat3.fitd(i, :), pat6.all_data(i,:));
    corr.threesix(i) = correlation(2,1);
end
%threeseven
for i = 1:seven
    for j = 1:4296 %lesser data points
        pat7.fitd(i,j) = pat7.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat7.fitd(i, :), pat3.all_data(i,:));
    corr.threeseven(i) = correlation(2,1);
end
%threeeight
for i = 1:seven
    for j = 1:2118 %lesser data points
        pat3.fite(i,j) = pat3.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat3.fite(i, :), pat8.all_data(i,:));
    corr.threeeight(i) = correlation(2,1);
end
%threenine
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat3.fite(i,j) = pat3.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat3.fite(i, :), pat9.all_data(i,:));
    corr.threenine(i) = correlation(2,1);
end
%threenine
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat3.fite(i,j) = pat3.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat3.fite(i, :), pat9.all_data(i,:));
    corr.threenine(i) = correlation(2,1);
end
%fourfive
for i = 1:seven
    for j = 1:3010 %lesser data points
        pat5.fite(i,j) = pat5.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat5.fite(i, :), pat4.all_data(i,:));
    corr.fourfive(i) = correlation(2,1);
end
%foursix
for i = 1:seven
    for j = 1:3010 %lesser data points
        pat6.fite(i,j) = pat6.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat6.fite(i, :), pat4.all_data(i,:));
    corr.foursix(i) = correlation(2,1);
end
%fourseven
for i = 1:seven
    for j = 1:3010 %lesser data points
        pat7.fite(i,j) = pat7.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat7.fite(i, :), pat4.all_data(i,:));
    corr.fourseven(i) = correlation(2,1);
end
%foureight
for i = 1:seven
    for j = 1:2118 %lesser data points
        pat4.fite(i,j) = pat4.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat4.fite(i, :), pat8.all_data(i,:));
    corr.foureight(i) = correlation(2,1);
end
%fournine
for i = 1:seven
    for j = 1:3010 %lesser data points
        pat9.fite(i,j) = pat9.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat9.fite(i, :), pat4.all_data(i,:));
    corr.fournine(i) = correlation(2,1);
end
%fivesix
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat5.fite(i,j) = pat5.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat5.fite(i, :), pat6.all_data(i,:));
    corr.fivesix(i) = correlation(2,1);
end
%fiveseven
for i = 1:seven
    for j = 1:4298 %lesser data points
        pat5.fite(i,j) = pat5.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat5.fite(i, :), pat7.all_data(i,:));
    corr.fiveseven(i) = correlation(2,1);
end
%fiveeight
for i = 1:seven
    for j = 1:2118 %lesser data points
        pat5.fitf(i,j) = pat5.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat5.fitf(i, :), pat8.all_data(i,:));
    corr.fiveeight(i) = correlation(2,1);
end
%fivenine
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat5.fitf(i,j) = pat5.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat5.fitf(i, :), pat9.all_data(i,:));
    corr.fivenine(i) = correlation(2,1);
end
%sixseven
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat7.fitf(i,j) = pat7.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat7.fitf(i, :), pat6.all_data(i,:));
    corr.sixseven(i) = correlation(2,1);
end
%sixeight
for i = 1:seven
    for j = 1:2118 %lesser data points
        pat6.fitf(i,j) = pat6.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat6.fitf(i, :), pat8.all_data(i,:));
    corr.sixeight(i) = correlation(2,1);
end
%sixnine
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat6.fitf(i,j) = pat6.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat6.fitf(i, :), pat9.all_data(i,:));
    corr.sixnine(i) = correlation(2,1);
end
%seveneight
for i = 1:seven
    for j = 1:2118 %lesser data points
        pat7.fitg(i,j) = pat7.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat7.fitg(i, :), pat8.all_data(i,:));
    corr.seveneight(i) = correlation(2,1);
end
%sevennine
for i = 1:seven
    for j = 1:4273 %lesser data points
        pat7.fitg(i,j) = pat7.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat7.fitg(i, :), pat9.all_data(i,:));
    corr.sevennine(i) = correlation(2,1);
end
%eightnine
for i = 1:seven
    for j = 1:2118 %lesser data points
        pat9.fitg(i,j) = pat9.all_data(i,j);
    end
end
for i = 1:seven
    correlation = corrcoef(pat9.fitg(i, :), pat8.all_data(i,:));
    corr.eightnine(i) = correlation(2,1);
end

% (b) We can see that patient 6 and 9 have the exact same data and are therefore
%faulty.



% (c) The function, corrcoef gives us the correlation coefficient. This function
% returns the matrix of correlation coefficients for each observation. Via
% this function we can clearly tell that patients 6 and 9 are the exact
% same data set since the corrcoef yields a matrix of all 1's.




%% Task2.2


ml_error = zeros(1,7);
map_error = zeros(1,7);
    for j = 1:7
        for i = 1:9
            temp1 = cell2mat(Error_table_array{i,j});
            ml_error(1,j) = ml_error(1,j) + temp1(1,3);
            temp2 = cell2mat(Error_table_array{i,j});
            map_error(1,j) = map_error(1,j) + temp2(2,3);
        end
    end

    ml_error_pat = zeros(1,9);
    map_error_pat = zeros(1,9);
    for i = 1:9
        for j = 1:7
            temp1 = cell2mat(Error_table_array{i,j});
            ml_error_pat(1,i) = ml_error_pat(1,i) + temp1(1,3);
            temp2 = cell2mat(Error_table_array{i,j});
            map_error_pat(1,i) = map_error_pat(1,i) + temp2(2,3);
        end
    end



sorted_ml_error = sort(ml_error);
sorted_map_error = sort(map_error);

sorted_ml_error_pat = sort(ml_error_pat);
sorted_map_error_pat = sort(map_error_pat);

first_best_pat_ml = find(ml_error_pat == sorted_ml_error_pat(1,1));
second_best_pat_ml = find(ml_error_pat == sorted_ml_error_pat(1,2));

first_best_pat_map = find(map_error_pat == sorted_map_error_pat(1,1));
second_best_pat_map = find(map_error_pat == sorted_map_error_pat(1,2));

fprintf('first_best_pat_ml = %d, second_best_pat_ml = %d\n',first_best_pat_ml,second_best_pat_ml  );
fprintf('first_best_pat_map = %d, second_best_pat_map = %d\n', first_best_pat_map, second_best_pat_map);

first_best_feature_ml = find(ml_error == sorted_ml_error(1,1));
second_best_feature_ml = find(ml_error == sorted_ml_error(1,2));
fprintf('first_best_feature_ml %d, second_best_feature_ml %d \n',first_best_feature_ml, second_best_feature_ml);
first_best_feature_map = find(map_error == sorted_map_error(1,1));
second_best_feature_map = find(map_error == sorted_map_error(1,2));
third_best_feature_map = find(map_error == sorted_map_error(1,3));
fprintf('first_best_feature_map %d, second_best_feature_map %d, %d \n',first_best_feature_map, second_best_feature_map,third_best_feature_map);

corr_cell_array = cell(9,7);
for j = 1:7
    for i = 1:9
        corr_cell_array(i,j) = mat2cell(zeros(1,7),1,7);
    end
end

for p = 1:9
    for f1 = 1:7
        for f2 = 1:7
           cor = corrcoef(testing(p).all_data(f1,:), testing(p).all_data(f2,:));
           if cor(1,2) < 0.1 && cor(1,2) > -0.1
            corr_cell_array{p,f1}(1,f2) = 1;
           end
        end
    end
end


%fclose(fid);
%% Task 3.1a
% Takes the likelihood matrices for the selected pairs of features in Task 2.1
%(saved in the HT_table_array from task 1.1, part e), to generate
% the likelihood matrices for joint observations from the pair of features
% Looking at Patients 3, 6, and 7, Features 1 and 6
% Looking at Patients 6, 7, and 8, Features 2 and 3
%Patient 6
%%%%%%%%%%%%%%%%%%%
firstpatnumber = 6;
secondpatnumber = 7;
thirdpatnumber = 8;
firstfeature = 1;
secondfeature = 6;
%%%%%%%%%%%%%%%%%%
firstIt = size(HT_table_array{firstpatnumber,firstfeature});
secondIt = size(HT_table_array{firstpatnumber,secondfeature});
for i=1:firstIt(1)
    for j=1:secondIt(1)
        val1 = HT_table_array{firstpatnumber,firstfeature}(i,2);
        val2 = HT_table_array{firstpatnumber,secondfeature}(j,2);
        val3 = HT_table_array{firstpatnumber,firstfeature}(i,3);
        val4 = HT_table_array{firstpatnumber,secondfeature}(j,3);
        H1_1((i-1)*secondIt(1)+j) = val1*val2;
        H0_1((i-1)*secondIt(1)+j) = val3*val4;
    end
end
likelihoodmatr1 = vertcat(H1_1, H0_1);
%Patient 7

firstIt = size(HT_table_array{secondpatnumber,firstfeature});
secondIt = size(HT_table_array{secondpatnumber,secondfeature});
for i=1:firstIt(1)
    for j=1:secondIt(1)
        val1 = HT_table_array{secondpatnumber,firstfeature}(i,2);
        val2 = HT_table_array{secondpatnumber,secondfeature}(j,2);
        val3 = HT_table_array{secondpatnumber,firstfeature}(i,3);
        val4 = HT_table_array{secondpatnumber,secondfeature}(j,3);
        H1_2((i-1)*secondIt(1)+j) = val1*val2;
        H0_2((i-1)*secondIt(1)+j) = val3*val4;
    end
end
likelihoodmatr2 = vertcat(H1_2, H0_2);

%Patient8

firstIt = size(HT_table_array{thirdpatnumber,firstfeature});
secondIt = size(HT_table_array{thirdpatnumber,secondfeature});
for i=1:firstIt(1)
    for j=1:secondIt(1)
        val1 = HT_table_array{thirdpatnumber,firstfeature}(i,2);
        val2 = HT_table_array{thirdpatnumber,secondfeature}(j,2);
        val3 = HT_table_array{thirdpatnumber,firstfeature}(i,3);
        val4 = HT_table_array{thirdpatnumber,secondfeature}(j,3);
        H1_3((i-1)*secondIt(1)+j) = val1*val2;
        H0_3((i-1)*secondIt(1)+j) = val3*val4;
    end
end
likelihoodmatr3 = vertcat(H1_3, H0_3);

%b. Calculates the ML and MAP decision rule vectors, based on the likelihood matrix
% of the joint observations from the selected pair. To be definite, break ties in favor of H1.
%Pat6
for i=1:length(likelihoodmatr1)
    if(likelihoodmatr1(2*i-1) >= likelihoodmatr1(2*i))
        ML_p1(i) = 1;
    else
        ML_p1(i) = 0;
    end
end
for i=1:length(likelihoodmatr1)
    if(likelihoodmatr1(2*i-1)*P_H1(1) >= likelihoodmatr1(2*i)*P_H0(1))
        MAP_p1(i) = 1;
    else
        MAP_p1(i) = 0;
    end
end
%Pat2
for i=1:length(likelihoodmatr2)
    if(likelihoodmatr2(2*i-1) >= likelihoodmatr2(2*i))
        ML_p2(i) = 1;
    else
        ML_p2(i) = 0;
    end
end
for i=1:length(likelihoodmatr2)
    if(likelihoodmatr2(2*i-1)*P_H1(2) >= likelihoodmatr2(2*i)*P_H0(2))
        MAP_p2(i) = 1;
    else
        MAP_p2(i) = 0;
    end
end
%Pat3
for i=1:length(likelihoodmatr3)
    if(likelihoodmatr3(2*i-1) >= likelihoodmatr3(2*i))
        ML_p3(i) = 1;
    else
        ML_p3(i) = 0;
    end
end
for i=1:length(likelihoodmatr3)
    if(likelihoodmatr3(2*i-1)*P_H1(3) >= likelihoodmatr3(2*i)*P_H0(3))
        MAP_p3(i) = 1;
    else
        MAP_p3(i) = 0;
    end
end

%c
% Pat3:
size_4 = size(HT_table_array{firstpatnumber,firstfeature});
size_5 = size(HT_table_array{firstpatnumber,secondfeature});
for i=1:size_4(1)
    for j=1:size_5(1)
        col1_1((i-1)*size_5(1)+j) = min(HT_table_array{3,1}(:,1)+i-1);
        col2_1((i-1)*size_5(1)+j) = min(HT_table_array{3,6}(:,1)+j-1);
    end
end
col1_1T = col1_1';
col2_1T = col2_1';
col3_1 = transpose(likelihoodmatr1);
col4_1 = transpose(ML_p1);
col5_1 = transpose(MAP_p1);
Joint_HT_table_1 = horzcat(col1_1T, col2_1T, col3_1, col4_1, col5_1);

%Pat6
size_4 = size(HT_table_array{secondpatnumber,firstfeature});
size_5 = size(HT_table_array{secondpatnumber,secondfeature});
for i=1:size_4(1)
    for j=1:size_5(1)
        col1_2((i-1)*size_5(1)+j) = min(HT_table_array{6,1}(:,1)+i-1);
        col2_2((i-1)*size_5(1)+j) = min(HT_table_array{6,6}(:,1)+j-1);
    end
end
col1_2T = col1_2';
col2_2T = col2_2';
col3_2 = transpose(likelihoodmatr2);
col4_2 = transpose(ML_p2);
col5_2 = transpose(MAP_p2);
Joint_HT_table_2 = horzcat(col1_2T, col2_2T, col3_2, col4_2, col5_2);

% Patient 7:
size_4 = size(HT_table_array{thirdpatnumber,firstfeature});
size_5 = size(HT_table_array{thirdpatnumber,secondfeature});
for i=1:size_4(1)
    for j=1:size_5(1)
        col1_3((i-1)*size_5(1)+j) = min(HT_table_array{7,1}(:,1)+i-1);
        col2_3((i-1)*size_5(1)+j) = min(HT_table_array{7,6}(:,1)+j-1);
    end
end
col1_3T = col1_3';
col2_3T = col2_3';
col3_3 = transpose(likelihoodmatr3);
col4_3 = transpose(ML_p3);
col5_3 = transpose(MAP_p3);
Joint_HT_table_3 = horzcat(col1_3T, col2_3T, col3_3, col4_3, col5_3);

%d
%First pat
X = HT_table_array{firstpatnumber,firstfeature}(:,1);
Y = HT_table_array{firstpatnumber,secondfeature}(:,1);
Z = vec2mat(Joint_HT_table_1(:,3), length(X));
mesh(X,Y,Z);
figure;

Z = vec2mat(Joint_HT_table_1(:,4), length(X));
mesh(X,Y,Z);
figure;

%second pat
X = HT_table_array{secondpatnumber,firstfeature}(:,1);
Y = HT_table_array{secondpatnumber,secondfeature}(:,1);
Z = vec2mat(Joint_HT_table_2(:,3), length(X));
mesh(X,Y,Z);
figure;

Z = vec2mat(Joint_HT_table_2(:,4), length(X));
mesh(X,Y,Z);
figure;

%thirdpat
X = HT_table_array{thirdpatnumber,firstfeature}(:,1);
Y = HT_table_array{thirdpatnumber,secondfeature}(:,1);
Z = vec2mat(Joint_HT_table_3(:,3), length(X));
mesh(X,Y,Z);
figure;

Z = vec2mat(Joint_HT_table_3(:,4), length(X));
mesh(X,Y,Z);
figure;


%% Task 3.2

% Patient 6
% The following code attempts to generate alarms based on
% each of the ML and MAP decision rules for the testing
% data set provided to us for the selected patient, i.e.
% patient six

count = testing_data_points(6);
value_ml = zeros(1, count);
value_map = zeros(1, count);

for i=1:count

  var1 = testing(6).all_data(1, i);
  var2 = testing(6).all_data(6, i);

  [ind1, col1] = find(Joint_HT_table_1(:, 1) == var1);

  %% used for error checking

  if (size(ind1, 1) == 0 || size(col1, 1) == 0)
      message_1 = 'Could not find index1';
  end

  lab1 = Joint_HT_table_1(ind1, :);

  [ind2, col2] = find(lab1(:, 2) == var2);

  if (size(ind2, 1) == 0 || size(col2, 1) == 0)
      message_1 = 'Could not find index2';
  end

  lab2 = lab1(ind2, :);

  if (size(lab2) ~= 0)
      value_ml(i) = lab2(1, 5);
      value_map(i) = lab2(1, 6);

  elseif (size(lab2) == 0)
      value_ml(i) = 0;
      value_map(i) = 0;

  else
      continue;

  end

end

fa_map = 0;
md_map = 0;
e_map = 0;

fa_ml = 0;
md_ml = 0;
e_ml = 0;

 for i=1:count

  if (testing(6).all_labels(i) == 0 && value_ml(i) == 1)
      fa_ml = fa_ml + 1;

  elseif (testing(6).all_labels(i) == 1 && value_ml(i) == 0)
      md_ml = md_ml + 1;
  end

  boolean3 = testing(6).all_labels(i) == 0 && value_ml(i) == 1;
  boolean4 = testing(6).all_labels(i) == 1 && value_ml(i) == 0;

  if (boolean3 || boolean4)
      e_ml = e_ml + 1;
  end

  %% MAP

  if (testing(6).all_labels(i) == 0 && value_map(i) == 1)
      fa_map = fa_map + 1;

  elseif (testing(6).all_labels(i) == 1 && value_map(i) == 0)
      md_map = md_map + 1;
  end

  boolean1 = testing(6).all_labels(i) == 0 && value_map(i) == 1;
  boolean2 = testing(6).all_labels(i) == 1 && value_map(i) == 0;

  if (boolean1 || boolean2)
      e_map = e_map + 1;
  end

end

patient1_cell = zeros(2, 3);
patient1_h1 = sum(testing(6).all_labels);
patient1_h0 = count - patient1_h1;

value_array = [fa_map/patient1_h0, md_map/patient1_h1, e_map/count, fa_ml/patient1_h0, md_ml/patient1_h1, e_ml/count];

for i=1:3
  patient1_cell(1, i) = value_array(i);
end

for j=1:3
  patient1_cell(2, j) = value_array(j+3);
end

% Patient 7

% The following code attempts to generate alarms based on
% each of the ML and MAP decision rules for the testing
% data set provided to us for the selected patient, i.e.
% patient seven

count1 = testing_data_points(7);
value_ml1 = zeros(1, count1);
value_map1 = zeros(1, count1);

for i=1:count1

  var1 = testing(7).all_data(1, i);
  var2 = testing(7).all_data(6, i);

  [ind1, col1] = find(Joint_HT_table_2(:, 1) == var1);

  %% used for error checking

  if (size(ind1, 1) == 0 || size(col1, 1) == 0)
      message_1 = 'Could not find index1';
  end

  lab1 = Joint_HT_table_2(ind1, :);

  [ind2, col2] = find(lab1(:, 2) == var2);

  if (size(ind2, 1) == 0 || size(col2, 1) == 0)
      message_1 = 'Could not find index2';
  end

  lab2 = lab1(ind2, :);

   if (size(lab2) ~= 0)
      value_ml1(i) = lab2(1, 5);
      value_map1(i) = lab2(1, 6);

  elseif (size(lab2) == 0)
      value_ml1(i) = 0;
      value_map1(i) = 0;
  else
      continue;

   end
end

fa_map = 0;
md_map = 0;
e_map = 0;

fa_ml = 0;
md_ml = 0;
e_ml = 0;

  %% ML

 for i=1:count1

  if (testing(7).all_labels(i) == 0 && value_ml1(i) == 1)
      fa_ml = fa_ml + 1;

  elseif (testing(7).all_labels(i) == 1 && value_ml1(i) == 0)
      md_ml = md_ml + 1;

  end

  boolean3 = testing(7).all_labels(i) == 0 && value_ml1(i) == 1;
  boolean4 = testing(7).all_labels(i) == 1 && value_ml1(i) == 0;

  if (boolean3 || boolean4)
      e_ml = e_ml + 1;
  end

  %% MAP

  if (testing(7).all_labels(i) == 0 && value_map1(i) == 1)
      fa_map = fa_map + 1;

  elseif (testing(7).all_labels(i) == 1 && value_map1(i) == 0)
      md_map = md_map + 1;
  end

  boolean1 = testing(7).all_labels(i) == 0 && value_map1(i) == 1;
  boolean2 = testing(7).all_labels(i) == 1 && value_map1(i) == 0;

  if (boolean1 || boolean2)
      e_map = e_map + 1;
  end

end

patient2_cell = zeros(2, 3);
patient2_h1 = sum(testing(7).all_labels);
patient2_h0 = count1 - patient2_h1;

value_array = [fa_map/patient2_h0, md_map/patient2_h1, e_map/count1, fa_ml/patient2_h0, md_ml/patient2_h1, e_ml/count1];

for i=1:3
  patient2_cell(1, i) = value_array(i);
end

for j=1:3
  patient2_cell(2, j) = value_array(j+3);
end

% Patient 8

% The following code attempts to generate alarms based on
% each of the ML and MAP decision rules for the testing
% data set provided to us for the selected patient, i.e.
% patient eight

count2 = testing_data_points(8);
value_ml2 = zeros(1, count2);
value_map2 = zeros(1, count2);

for i=1:count2

  var1 = testing(8).all_data(1, i);
  var2 = testing(8).all_data(6, i);

  [ind1, col1] = find(Joint_HT_table_3(:, 1) == var1);

  %% used for error checking

  if (size(ind1, 1) == 0 || size(col1, 1) == 0)
      message_1 = 'Could not find index1';
  end

  lab1 = Joint_HT_table_3(ind1, :);

  [ind2, col2] = find(lab1(:, 2) == var2);

  if (size(ind2, 1) == 0 || size(col2, 1) == 0)
      message_1 = 'Could not find index2';
  end

  lab2 = lab1(ind2, :);

   if (size(lab2) ~= 0)
      value_ml2(i) = lab2(1, 5);
      value_map2(i) = lab2(1, 6);

  elseif (size(lab2) == 0)
      value_ml2(i) = 0;
      value_map2(i) = 0;
  else
      continue;

   end

end

fa_map = 0;
md_map = 0;
e_map = 0;

fa_ml = 0;
md_ml = 0;
e_ml = 0;

for i=1:count2
  %% ML

  if (testing(8).all_labels(i) == 0 && value_ml2(i) == 1)
      fa_ml = fa_ml + 1;

  elseif (testing(8).all_labels(i) == 1 && value_ml2(i) == 0)
      md_ml = md_ml + 1;
  end

  boolean3 = testing(8).all_labels(i) == 0 && value_ml2(i) == 1;
  boolean4 = testing(8).all_labels(i) == 1 && value_ml2(i) == 0;

  if (boolean3 || boolean4)
      e_ml = e_ml + 1;
  end

  %% MAP

  if (testing(8).all_labels(i) == 0 && value_map2(i) == 1)
      fa_map = fa_map + 1;

  elseif (testing(8).all_labels(i) == 1 && value_map2(i) == 0)
      md_map = md_map + 1;
  end

  boolean1 = testing(8).all_labels(i) == 0 && value_map2(i) == 1;
  boolean2 = testing(8).all_labels(i) == 1 && value_map2(i) == 0;

  if (boolean1 || boolean2)
      e_map = e_map + 1;
  end

end

patient3_cell = zeros(2, 3);
patient3_h1 = sum(testing(8).all_labels);
patient3_h0 = count2 - patient3_h1;

value_array = [fa_map/patient3_h0, md_map/patient3_h1, e_map/count2, fa_ml/patient3_h0, md_ml/patient3_h1, e_ml/count2];

for i=1:3
  patient3_cell(1, i) = value_array(i);
end

for j=1:3
  patient3_cell(2, j) = value_array(j+3);
end

%% (c)
%% Plotting the graphs below

% Plotting the alarms generated based on testing data set for a pair of features

figure;

subplot(3, 1, 1);
bar(value_map);
title(['Patient 6', ' MAP Alarms']);

subplot(3, 1, 2);
bar(value_ml);
title(['Patient 6', ' ML Alarms']);

subplot(3, 1, 3);
bar(testing(6).all_labels);
title(['Patient 6', ' Golden Alarms']);

figure;

subplot(3, 1, 1);
bar(value_map1);
title(['Patient 7', ' MAP Alarms']);

subplot(3, 1, 2);
bar(value_ml1);
title(['Patient 7', ' ML Alarms']);

subplot(3, 1, 3);
bar(testing(7).all_labels);
title(['Patient 7', ' Golden Alarms']);

figure;

subplot(3, 1, 1);
bar(value_map2);
title(['Patient 8', ' MAP Alarms']);

subplot(3, 1, 2);
bar(value_ml2);
title(['Patient 8', ' ML Alarms']);

subplot(3, 1, 3);
bar(testing(8).all_labels);
title(['Patient 8', ' Golden Alarms']);

%% Task 3.3

% Calculating average probability of error for each of the ML and MAP
% achieved for the selected 3 patients

sum_ML_error = patient1_cell(1, 3) + patient2_cell(1, 3) + patient3_cell(1, 3);
sum_MAP_error = patient1_cell(2, 3) + patient2_cell(2, 3) + patient3_cell(2, 3);

average_ML_error = (sum_ML_error)/(3);
average_MAP_error = (sum_MAP_error)/(3);

% %% Task 3.4
% %
% % feature_four_patient6 = testing(6).all_data(4,:);
% % feature_five_patient6 = testing(6).all_data(5,:);
% %
% % feature_four_patient7 = testing(7).all_data(4,:);
% % feature_five_patient7 = testing(7).all_data(5,:);
% %
% % feature_four_patient8 = testing(8).all_data(4,:);
% % feature_five_patient8 = testing(8).all_data(5,:);
% %
% % for i = 1:size(feature_four_patient6,2) j = 1:size(feature_five_patient6,2)
% %
% %
% %
% % end
%
%
% %Creating golden and non golden parts for the testing data set
% for i = 1:9
%
%     j = 1;
%
%     while (j < 8)
%
%         golden = 1;                                                                           % keep track of indices
%         non_golden = 1;
%
%         for k = 1:size(testing(i).all_data,2)
%
%             if (testing(i).all_labels(1, k))
%                 testing(i).golden_data(j, golden) = testing(i).all_data(j, k);
%                 golden = golden + 1;                                                          % increment index
%
%             else
%                 testing(i).non_golden_data(j, non_golden) = testing(i).all_data(j, k);
%                 non_golden = non_golden + 1;
%             end
%
%         end
%
%      j = j+1;
%
%     end
%
% end
%
% %Patient 6, feature 1 and 6
%
% %Finding displacement for patient 6 and feature 1
% golden = testing(6).golden_data(1,:);
% gold_tab = tabulate(floor(golden))';
% nongolden = testing(6).non_golden_data(1,:);
% nongolden_tab = tabulate(floor(nongolden))';
% min_gold_tab = min(gold_tab(1,:));
% max_gold_tab = max(gold_tab(1,:));
% min_nongolden_tab = min(nongolden_tab(1,:));
% max_nongolden_tab = max(nongolden_tab(1,:));
% lesser_range = min(min_gold_tab, min_nongolden_tab);
% max_range = max(max_gold_tab,max_nongolden_tab);
% displacement_1x = 0 - lesser_range + 1;
%
% %Finding displacement for patient 6 and feature 6
% golden = testing(6).golden_data(6,:);
% gold_tab = tabulate(floor(golden))';
% nongolden = testing(6).non_golden_data(6,:);
% nongolden_tab = tabulate(floor(nongolden))';
% min_gold_tab = min(gold_tab(1,:));
% max_gold_tab = max(gold_tab(1,:));
% min_nongolden_tab = min(nongolden_tab(1,:));
% max_nongolden_tab = max(nongolden_tab(1,:));
% lesser_range = min(min_gold_tab, min_nongolden_tab);
% max_range = max(max_gold_tab,max_nongolden_tab);
% displacement_1y = 0 - lesser_range + 1;
%
%
%
% x_array_six = Joint_HT_table_1(:,1);
% y_array_six = Joint_HT_table_1(:,2);
%
% test_val_one = testing(6).all_data(1,4);
% fprintf('Value of one : %d\n',test_val_one);
% test_val_two = testing(6).all_data(6,4);
% fprintf('Value of two : %d\n', test_val_two);
%
% temp1 = find(x_array_six == test_val_one);
% temp2 = find(y_array_six == test_val_two);
% if ((size(temp1,1) == 0) || (size(temp2,1) == 0))
%     fprintf('Not in x_array or y_array\n');
% end
% % row_index_p_h1 = temp1(1,1) + temp2(1,1) - 1;
% %
% % p_h1 = Joint_HT_table_1(row_index_p_h1, 4);
%
% for i = 1:size(testing(6).all_data,2)
%     test_val_one = testing(6).all_data(1,i);
%     test_val_two = testing(6).all_data(6,i);
%     test_val_one = test_val_one + displacement_1x;
%     test_val_two = test_val_two + displacement_1y;
%     temp1 = find(x_array_six == test_val_one);
%     temp2 = find(y_array_six == test_val_two);
%     if ((size(temp1,1) == 0) || (size(temp2,1) == 0))
%         fprintf('Value of i %d\n',i);
%         fprintf('temp1 or temp2 is empty\n');
%         p_h1_six(i) = 0;
%         fprintf('Value of test1 = %d, test2 = %d\n',test_val_one,test_val_two);
%     else
%         row_index_p_h1 = temp1(1,1) + temp2(1,1) - 1;
%         p_h1_six(i) = Joint_HT_table_1(row_index_p_h1, 3);
%     end
% end
%
% figure;
%
% pos = 1
% [x,y] = perfcurve(testing(6).all_labels,p_h1_six, 1);
% plot(x,y);
% xlabel('False positive rate');
% ylabel( 'True positive rate')
% % For patient 7, features 1 and 6
%
% %Finding displacement for patient 7 and features 1
% golden = testing(7).golden_data(1,:);
% gold_tab = tabulate(floor(golden))';
% nongolden = testing(7).non_golden_data(1,:);
% nongolden_tab = tabulate(floor(nongolden))';
% min_gold_tab = min(gold_tab(1,:));
% max_gold_tab = max(gold_tab(1,:));
% min_nongolden_tab = min(nongolden_tab(1,:));
% max_nongolden_tab = max(nongolden_tab(1,:));
% lesser_range = min(min_gold_tab, min_nongolden_tab);
% max_range = max(max_gold_tab,max_nongolden_tab);
% displacement_2x = 0 - lesser_range + 1;
% fprintf('Value of displacement_2x %d\n', displacement_2x);
% %Finding displacement for patient 7 and feature 6
% golden = testing(7).golden_data(6,:);
% gold_tab = tabulate(floor(golden))';
% nongolden = testing(7).non_golden_data(6,:);
% nongolden_tab = tabulate(floor(nongolden))';
% min_gold_tab = min(gold_tab(1,:));
% max_gold_tab = max(gold_tab(1,:));
% min_nongolden_tab = min(nongolden_tab(1,:));
% max_nongolden_tab = max(nongolden_tab(1,:));
% lesser_range = min(min_gold_tab, min_nongolden_tab);
% fprintf('Value of lesser_range %d\n', lesser_range);
% max_range = max(max_gold_tab,max_nongolden_tab);
% displacement_2y = 0 - lesser_range + 1;
% fprintf('Value of displacement_2y %d\n', displacement_2y);
%
% x_array_seven = Joint_HT_table_2(:,1);
% y_array_seven = Joint_HT_table_2(:,2);
%
% for i = 1:size(testing(7).all_data,2)
%     test_val_one = testing(7).all_data(1,i);
%     test_val_two = testing(7).all_data(6,i);
%     test_val_one = test_val_one + displacement_2x;
%     test_val_two = test_val_two + displacement_2y;
%     temp1 = find(x_array_seven == test_val_one);
%     temp2 = find(y_array_seven == test_val_two);
%     if ((size(temp1,1) == 0) || (size(temp2,1) == 0))
%         fprintf('temp1 or temp2 is empty patient 7 test_val_one %d test_val_two %d\n', test_val_one, test_val_two);
%         p_h1_seven(i) = 0;
%
%     else
%         row_index_p_h1 = temp1(1,1) + temp2(1,1) - 1;
%          p_h1_seven(i) = Joint_HT_table_2(row_index_p_h1, 3);
%     end
% end
%         fprintf('got after patient 7\n');
%
% figure;
% pos = 1
% [x,y] = perfcurve(testing(7).all_labels,p_h1_seven, 1);
% plot(x,y);
% xlabel('False positive rate');
% ylabel( 'True positive rate')
%
% % For patient 8, features 1 and 6
%
% %Finding displacement for patient 8 and features 1
% golden = testing(8).golden_data(1,:);
% gold_tab = tabulate(floor(golden))';
% nongolden = testing(8).non_golden_data(1,:);
% nongolden_tab = tabulate(floor(nongolden))';
% min_gold_tab = min(gold_tab(1,:));
% max_gold_tab = max(gold_tab(1,:));
% min_nongolden_tab = min(nongolden_tab(1,:));
% max_nongolden_tab = max(nongolden_tab(1,:));
% lesser_range = min(min_gold_tab, min_nongolden_tab);
% max_range = max(max_gold_tab,max_nongolden_tab);
% displacement_3x = 0 - lesser_range + 1;
%
% %Finding displacement for patient 7 and feature 6
% golden = testing(8).golden_data(6,:);
% gold_tab = tabulate(floor(golden))';
% nongolden = testing(8).non_golden_data(6,:);
% nongolden_tab = tabulate(floor(nongolden))';
% min_gold_tab = min(gold_tab(1,:));
% max_gold_tab = max(gold_tab(1,:));
% min_nongolden_tab = min(nongolden_tab(1,:));
% max_nongolden_tab = max(nongolden_tab(1,:));
% lesser_range = min(min_gold_tab, min_nongolden_tab);
% max_range = max(max_gold_tab,max_nongolden_tab);
% displacement_3y = 0 - lesser_range + 1;
%
%
% x_array_eight = Joint_HT_table_3(:,1);
% y_array_eight = Joint_HT_table_3(:,2);
%
% for i = 1:size(testing(8).all_data,2)
%     test_val_one = testing(8).all_data(1,i);
%     test_val_two = testing(8).all_data(6,i);
%     test_val_one = test_val_one + displacement_3x;
%     test_val_two = test_val_two + displacement_3y;
%     temp1 = find(x_array_eight == test_val_one);
%     temp2 = find(y_array_eight == test_val_two);
%     if ((size(temp1,1) == 0) || (size(temp2,1) == 0))
%         fprintf('temp1 or temp2 is empty patient 8 test_val_one %d test_val_two %d\n', test_val_one, test_val_two);
%         p_h1_eight(i) = 0;
%
%     else
%         row_index_p_h1 = temp1(1,1) + temp2(1,1) - 1;
%          p_h1_eight(i) = Joint_HT_table_3(row_index_p_h1, 3);
%     end
% end
%
%         fprintf('got after patient 8\n');
%
% %
% %
% %
% %
%
%
%
%
%
%
%
%
%
% % close all;
%% Task 3.4
%
% feature_four_patient6 = testing(6).all_data(4,:);
% feature_five_patient6 = testing(6).all_data(5,:);
%
% feature_four_patient7 = testing(7).all_data(4,:);
% feature_five_patient7 = testing(7).all_data(5,:);
%
% feature_four_patient8 = testing(8).all_data(4,:);
% feature_five_patient8 = testing(8).all_data(5,:);
%
% for i = 1:size(feature_four_patient6,2) j = 1:size(feature_five_patient6,2)
%
%
%
% end


%Creating golden and non golden parts for the testing data set
for i = 1:9

    j = 1;

    while (j < 8)

        golden = 1;                                                                           % keep track of indices
        non_golden = 1;

        for k = 1:size(testing(i).all_data,2)

            if (testing(i).all_labels(1, k))
                testing(i).golden_data(j, golden) = testing(i).all_data(j, k);
                golden = golden + 1;                                                          % increment index

            else
                testing(i).non_golden_data(j, non_golden) = testing(i).all_data(j, k);
                non_golden = non_golden + 1;
            end

        end

     j = j+1;

    end

end

%Patient 6, feature 1 and 6

%Finding displacement for patient 6 and feature 1
golden = testing(6).golden_data(1,:);
gold_tab = tabulate(floor(golden))';
nongolden = testing(6).non_golden_data(1,:);
nongolden_tab = tabulate(floor(nongolden))';
min_gold_tab = min(gold_tab(1,:));
max_gold_tab = max(gold_tab(1,:));
min_nongolden_tab = min(nongolden_tab(1,:));
max_nongolden_tab = max(nongolden_tab(1,:));
lesser_range = min(min_gold_tab, min_nongolden_tab);
max_range = max(max_gold_tab,max_nongolden_tab);
displacement_1x = 0 - lesser_range + 1;

%Finding displacement for patient 6 and feature 6
golden = testing(6).golden_data(6,:);
gold_tab = tabulate(floor(golden))';
nongolden = testing(6).non_golden_data(6,:);
nongolden_tab = tabulate(floor(nongolden))';
min_gold_tab = min(gold_tab(1,:));
max_gold_tab = max(gold_tab(1,:));
min_nongolden_tab = min(nongolden_tab(1,:));
max_nongolden_tab = max(nongolden_tab(1,:));
lesser_range = min(min_gold_tab, min_nongolden_tab);
max_range = max(max_gold_tab,max_nongolden_tab);
displacement_1y = 0 - lesser_range + 1;



x_array_six = Joint_HT_table_1(:,1);
y_array_six = Joint_HT_table_1(:,2);

% test_val_one = testing(6).all_data(1,4);
% fprintf('Value of one : %d\n',test_val_one);
% test_val_two = testing(6).all_data(6,4);
% fprintf('Value of two : %d\n', test_val_two);
%
% temp1 = find(x_array_six == test_val_one);
% temp2 = find(y_array_six == test_val_two);
% if ((size(temp1,1) == 0) || (size(temp2,1) == 0))
%     fprintf('Not in x_array or y_array\n');
% end
%
% row_index_p_h1 = temp1(1,1) + temp2(1,1) - 1;
%
% p_h1 = Joint_HT_table_1(row_index_p_h1, 4);

for i = 1:size(testing(6).all_data,2)
    test_val_one = testing(6).all_data(1,i);
    test_val_two = testing(6).all_data(6,i);
    test_val_one = test_val_one + displacement_1x;
    test_val_two = test_val_two + displacement_1y;
    temp1 = find(x_array_six == test_val_one);
    temp2 = find(y_array_six == test_val_two);
    if ((size(temp1,1) == 0) || (size(temp2,1) == 0))
 %       fprintf('Value of i %d\n',i);
 %       fprintf('temp1 or temp2 is empty\n');
        p_h1_six(i) = 0;
  %      fprintf('Value of test1 = %d, test2 = %d\n',test_val_one,test_val_two);
    else
        row_index_p_h1 = temp1(1,1) + temp2(1,1) - 1;
        p_h1_six(i) = Joint_HT_table_1(row_index_p_h1, 3);
    end
end

figure;

pos = 1
[x,y] = perfcurve(testing(6).all_labels,p_h1_six, 1);
plot(x,y);
xlabel('False positive rate');
ylabel( 'True positive rate')
% For patient 7, features 1 and 6

%Finding displacement for patient 7 and features 1
golden = testing(7).golden_data(1,:);
gold_tab = tabulate(floor(golden))';
nongolden = testing(7).non_golden_data(1,:);
nongolden_tab = tabulate(floor(nongolden))';
min_gold_tab = min(gold_tab(1,:));
max_gold_tab = max(gold_tab(1,:));
min_nongolden_tab = min(nongolden_tab(1,:));
max_nongolden_tab = max(nongolden_tab(1,:));
lesser_range = min(min_gold_tab, min_nongolden_tab);
max_range = max(max_gold_tab,max_nongolden_tab);
displacement_2x = 0 - lesser_range + 1;
%fprintf('Value of displacement_2x %d\n', displacement_2x);
%Finding displacement for patient 7 and feature 6
golden = testing(7).golden_data(6,:);
gold_tab = tabulate(floor(golden))';
nongolden = testing(7).non_golden_data(6,:);
nongolden_tab = tabulate(floor(nongolden))';
min_gold_tab = min(gold_tab(1,:));
max_gold_tab = max(gold_tab(1,:));
min_nongolden_tab = min(nongolden_tab(1,:));
max_nongolden_tab = max(nongolden_tab(1,:));
lesser_range = min(min_gold_tab, min_nongolden_tab);
%fprintf('Value of lesser_range %d\n', lesser_range);
max_range = max(max_gold_tab,max_nongolden_tab);
displacement_2y = 0 - lesser_range + 1;
%fprintf('Value of displacement_2y %d\n', displacement_2y);

x_array_seven = Joint_HT_table_2(:,1);
y_array_seven = Joint_HT_table_2(:,2);

for i = 1:size(testing(7).all_data,2)
    test_val_one = testing(7).all_data(1,i);
    test_val_two = testing(7).all_data(6,i);
    test_val_one = test_val_one + displacement_2x;
    test_val_two = test_val_two + displacement_2y;
    temp1 = find(x_array_seven == test_val_one);
    temp2 = find(y_array_seven == test_val_two);
    if ((size(temp1,1) == 0) || (size(temp2,1) == 0))
        %fprintf('temp1 or temp2 is empty patient 7 test_val_one %d test_val_two %d\n', test_val_one, test_val_two);
        p_h1_seven(i) = 0;
    else
        row_index_p_h1 = temp1(1,1) + temp2(1,1) - 1;
         p_h1_seven(i) = Joint_HT_table_2(row_index_p_h1, 3);
    end
end
        %fprintf('got after patient 7\n');

figure;
pos = 1
[x,y] = perfcurve(testing(7).all_labels,p_h1_seven, 1);
plot(x,y);
xlabel('False positive rate');
ylabel( 'True positive rate')

% For patient 8, features 1 and 6

%Finding displacement for patient 8 and features 1
golden = testing(8).golden_data(1,:);
gold_tab = tabulate(floor(golden))';
nongolden = testing(8).non_golden_data(1,:);
nongolden_tab = tabulate(floor(nongolden))';
min_gold_tab = min(gold_tab(1,:));
max_gold_tab = max(gold_tab(1,:));
min_nongolden_tab = min(nongolden_tab(1,:));
max_nongolden_tab = max(nongolden_tab(1,:));
lesser_range = min(min_gold_tab, min_nongolden_tab);
max_range = max(max_gold_tab,max_nongolden_tab);
displacement_3x = 0 - lesser_range + 1;

%Finding displacement for patient 7 and feature 6
golden = testing(8).golden_data(6,:);
gold_tab = tabulate(floor(golden))';
nongolden = testing(8).non_golden_data(6,:);
nongolden_tab = tabulate(floor(nongolden))';
min_gold_tab = min(gold_tab(1,:));
max_gold_tab = max(gold_tab(1,:));
min_nongolden_tab = min(nongolden_tab(1,:));
max_nongolden_tab = max(nongolden_tab(1,:));
lesser_range = min(min_gold_tab, min_nongolden_tab);
max_range = max(max_gold_tab,max_nongolden_tab);
displacement_3y = 0 - lesser_range + 1;


x_array_eight = Joint_HT_table_3(:,1);
y_array_eight = Joint_HT_table_3(:,2);

for i = 1:size(testing(8).all_data,2)
    test_val_one = testing(8).all_data(1,i);
    test_val_two = testing(8).all_data(6,i);
    test_val_one = test_val_one + displacement_3x;
    test_val_two = test_val_two + displacement_3y;
    temp1 = find(x_array_eight == test_val_one);
    temp2 = find(y_array_eight == test_val_two);
    if ((size(temp1,1) == 0) || (size(temp2,1) == 0))
       % fprintf('temp1 or temp2 is empty patient 8 test_val_one %d test_val_two %d\n', test_val_one, test_val_two);
        p_h1_eight(i) = 0;
    else
        row_index_p_h1 = temp1(1,1) + temp2(1,1) - 1;
         p_h1_eight(i) = Joint_HT_table_3(row_index_p_h1, 3);
    end
end
%fprintf('Value of displacement_3x %d, displacement_3y %d\n',displacement_3x,displacement_3y);
%fprintf('got after patient 8\n');
figure;
pos = 1
[x,y] = perfcurve(testing(8).all_labels,p_h1_eight, 1);
plot(x,y);
xlabel('False positive rate');
ylabel( 'True positive rate')














% close all;
