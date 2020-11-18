import sys, getopt

def levenshtein(word1,word2,is_damerau):
    len1 = len(word1)
    len2 = len(word2)

    matrix = [[0 for x in range(len1 + 1)] for y in range(len2 + 1)] 
    operation_matrix = [['' for x in range(len1 + 1)] for y in range(len2 + 1)]
   
    for i in range(1, len1 + 1):
        matrix[0][i] = i
        operation_matrix[0][i] += operation_matrix[0][i-1] + f'insert {word1[i-1]}, '
    for i in range(1, len2 + 1):
        matrix[i][0] = i
        operation_matrix[i][0] += operation_matrix[i-1][0] + f'delete {word2[i-1]}, '
    
    for i in range(1,len2 + 1):
        for j in range(1,len1 + 1):
            copy = matrix[i-1][j-1]
            delete = matrix[i-1][j]
            insert = matrix[i][j-1]

            if word1[j-1] == word2[i-1]:
                min_operation  = min(insert + 1 ,delete + 1, copy)
            else: min_operation = min(insert + 1 ,delete + 1, copy + 1)

            if (is_damerau) & (i > 1) & (j > 1) & (word1[j-2] == word2[i-1]) & (word1[j-1] == word2[i-2]):
                min_operation = min(min_operation, matrix[i-2][j-2] + 1)

            if min_operation == insert + 1:
                needed_operations = operation_matrix[i][j-1] + f'insert {word1[j-1]}, '
            elif min_operation == delete + 1:
                needed_operations = operation_matrix[i-1][j] + f'delete {word2[i-1]}, '
            elif (is_damerau) & (i > 1) & (j > 1) & (min_operation == matrix[i-2][j-2] + 1):
                needed_operations = operation_matrix[i-2][j-2] + f'transpose {word2[i-2]} and {word2[i-1]}, '
            elif word1[j-1] == word2[i-1]:
                needed_operations = operation_matrix[i-1][j-1] + f'copy {word2[i-1]}, '
            else:
                needed_operations = operation_matrix[i-1][j-1] + f'replace {word2[i-1]} to {word1[j-1]}, '
            
            matrix[i][j] = min_operation
            operation_matrix[i][j] = needed_operations 
    
    print(f'List of operations needed to trasform {word2} into {word1}: {operation_matrix[len2][len1][:-2]}')
    return matrix

def print_result(matrix,print_word1,print_word2,algorithm):
    len1 = len(print_word1) - 2
    len2 = len(print_word2) - 1

    print(f'{algorithm} edit distance is {matrix[len2][len1]}')
    
    for i in range(len(print_word1)):
        print(print_word1[i],end=' ')
    print()

    for i in range(len2 + 1):
        print(print_word2[i],end=' ')
        for j in range(len1 + 1):
            print(str(matrix[i][j]),end=' ')
        print()
    print()


word1 = sys.argv[2]
word2 = sys.argv[1]
len1 = len(word1)
len2 = len(word2)

print_word1 = '  ' + word1
print_word2 = ' ' + word2

matrix = levenshtein(word1,word2,False)

print_result(matrix,print_word1,print_word2,'Levenshtein')

matrix = levenshtein(word1,word2,True)

print_result(matrix,print_word1,print_word2,'Damerau-Levenshtein')

