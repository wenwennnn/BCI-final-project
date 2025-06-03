function recommendations = recommend_music(valence_input, arousal_input, csv1_path, csv2_path, top_n)
    % 讀取 CSV 檔案，只選取需要的欄位
    opts = detectImportOptions(csv1_path, 'VariableNamingRule', 'preserve');
    opts.SelectedVariableNames = {'song_id', 'valence_mean', 'arousal_mean'};
    T1 = readtable(csv1_path, opts);

    opts = detectImportOptions(csv2_path, 'VariableNamingRule', 'preserve');
    opts.SelectedVariableNames = {'song_id', 'valence_mean', 'arousal_mean'};
    T2 = readtable(csv2_path, opts);

    % 合併需要的欄位
    T = [T1; T2];

    % 去除欄位名稱空白
    T.Properties.VariableNames = strtrim(T.Properties.VariableNames);

    % 取出欄位
    valence = T.valence_mean;
    arousal = T.arousal_mean;
    song_id = T.song_id;

    % 移除 NaN
    valid_idx = ~isnan(valence) & ~isnan(arousal);
    valence = valence(valid_idx);
    arousal = arousal(valid_idx);
    song_id = song_id(valid_idx);

    % 計算距離
    distances = sqrt((valence - valence_input).^2 + (arousal - arousal_input).^2);
    [sorted_dist, idx] = sort(distances);
    top_idx = idx(1:top_n);

    % 回傳推薦
    recommendations = table(song_id(top_idx), valence(top_idx), arousal(top_idx), sorted_dist(1:top_n), ...
        'VariableNames', {'SongID', 'Valence', 'Arousal', 'Distance'});
end


% 設定 arousal 和 valence 輸入值
valence_input = 5.0;
arousal_input = 6.0;

% 設定檔案路徑（請確認這些檔案放在 MATLAB 當前資料夾下）
csv1 = '1.csv';
csv2 = '2.csv';

% 呼叫函數並取得推薦結果
recommendations = recommend_music(valence_input, arousal_input, csv1, csv2, 5);

% 顯示推薦結果
disp(recommendations)

