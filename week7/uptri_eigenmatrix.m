A = load('T5_A.txt');
my_eigenmatrix = load('T5_my_eigenmatrix.txt');

[V, D] = eig(A);    

n = size(V, 2);

for i = 1:n
    if V(i, i) ~= 0
        V(1:i, i) = V(1:i, i) / V(i, i);
    else
        warning('eigenmatrix(%d, %d) is zero', i, i);
    end
end

fileID = fopen('T5_matlab_eigenmatrix.txt', 'w');  % 打开文件进行写入

for i = 1:size(V, 1)
    fprintf(fileID, '%.6e ', V(i, :));
    fprintf(fileID, '\n');
end

fclose(fileID);