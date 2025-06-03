data = load('DREAMER.mat');
fieldnames(data)

class(data.DREAMER)           % 看它是什麼類型（例如 struct、cell、double...）
whos data                     % 看這個 struct 中所有欄位的大小與型別

fieldnames(data.DREAMER)

D = data.DREAMER.Data;

subject1 = D{1};                  % 取第一位受試者
fieldnames(subject1)             % 看有哪些欄位


D = data.DREAMER.Data;

subject1 = D{1};  % 取第1位受試者的 struct


baseline_data = subject1.EEG.baseline;
stimuli_data = subject1.EEG.stimuli;

class(baseline_data)
class(stimuli_data)

% 看 stimuli 裡面有幾個段落
length(subject1.EEG.stimuli)

% 看第 1 段是什麼類型
class(subject1.EEG.stimuli{1})

eeg_segment1 = subject1.EEG.stimuli{1};  % shape = [channels × samples]

% 查看大小
size(eeg_segment1)

% 畫第1通道前2000個時間點
eeg_segment1 = subject1.EEG.stimuli{1};  % [samples x channels]
eeg_segment1 = eeg_segment1';           % 轉置成 [channels x samples]

% 畫第1通道的前 2000 個點
plot(eeg_segment1(1, 1:2000));
title('Stimuli - Segment 1, Channel 1');
xlabel('Samples'); ylabel('Amplitude (μV)');

% 第1段的 EEG
eeg_segment1 = subject1.EEG.stimuli{1};
eeg_segment1 = eeg_segment1';

% 對應的情緒分數
valence1 = subject1.ScoreValence(1);
arousal1 = subject1.ScoreArousal(1);

% 畫圖加上情緒標籤
plot(eeg_segment1(1, 1:2000));
title(['Stimuli Segment 1, Channel 1 | Val: ', num2str(valence1), ', Aro: ', num2str(arousal1)]);
xlabel('Samples'); ylabel('Amplitude (μV)');

% 取出受試者
D = data.DREAMER.Data;
subject1 = D{1};

% 提取 valence 和 arousal 分數（共 18 筆 trial）
valence = subject1.ScoreValence;   % 18x1 double
arousal = subject1.ScoreArousal;   % 18x1 double

% 畫圖
figure;
plot(1:18, valence, '-o', 'LineWidth', 2); hold on;
plot(1:18, arousal, '-s', 'LineWidth', 2);
hold off;

% 設定圖示與標題
title('情緒變化折線圖（Subject 1）');
xlabel('Trial 編號');
ylabel('情緒分數 (1~9)');
legend({'Valence（愉悅程度）', 'Arousal（激動程度）'}, 'Location', 'best');
grid on;
