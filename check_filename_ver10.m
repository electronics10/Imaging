[file, path] = uigetfile('*.cfx');
cfx_filename = fullfile(path, file);
fid = fopen(cfx_filename);
assert(fid > -1);
prompt = {'Enter Nx here:','Enter Ny here:', 'Enter number of target:', 'File extension:'};
dlgtitle = 'Input';
dims = [1 30];
definput = {'51','51','1','s1p'};
info = inputdlg(prompt,dlgtitle,dims,definput);
[Nx, tf] = str2num(info{1, 1});
if ~tf
    disp('Nx should be a number!');
    return
end
if Nx < 1
    disp('Nx should be integer > 0');
    return
end
[Ny, tf] = str2num(info{2, 1});
if ~tf
    disp('Ny should be a number!');
    return
end
if Ny < 1
    disp('Ny should be integer > 0');
    return
end
[N, tf] = str2num(info{3, 1});
if ~tf
    disp('Ny should be a number!');
    return
end
if N < 1
    disp('Ny should be integer > 0');
    return
end
Nx = uint32(Nx);
Ny = uint32(Ny);
N = uint32(N);
TF = logical(false(1, Nx*Ny*N));
extension = strcat('*.', info{4, 1});
allfile = dir(fullfile(path, extension));
allfileName = {allfile.name};
id = strfind(cfx_filename, '.cfx');
identifier = extractBetween(cfx_filename, 1, id - 1);
pat = identifier{1, 1};
postfix = strcat('_SParameter1','.', info{4,1});
for i = 1:numel(allfileName)
    fullfileName = fullfile(path, allfileName{1, i});
    tf = contains(fullfileName, pat);
    if tf
        Number = str2double(extractBetween(fullfileName, strcat(pat, '_opt_'), postfix));
        if Number < 1
            disp('Conversion has error!')
            return
        end
        TF(1, Number) = true;
    end
end
F = find(TF == false);
disp("Loss file numbers in the list:");
disp(F);