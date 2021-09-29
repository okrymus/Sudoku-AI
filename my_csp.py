from sudoku import Sudoku
from copy import deepcopy

class CSP(object):
    '''
    This is a helper class to organize the problem that needs to be solved
    '''
    def __init__(self, variables, domains):
        '''

        :param variables: list of variables that need to be assigned
        :param domains: the possible values for each variable [(var, vals)]
        '''
        self.variables = variables
        self.domains = domains
        self.constraints = self.init_constraints()


    def init_constraints(self):
        '''
        Implement constraints.
        Define constraints such they are fast to evaluate. One method is to define the constraints as a
                function of the row and column. Sets are a fast way to remove duplicates.
        :return: constraints
        '''
        def cross(A, B):
            return [(int(a), int(b)) for a in A for b in B]

        cols = '012345678'
        rows = '012345678'

        list = ([cross(rows, c) for c in cols] +
                         [cross(r, cols) for r in rows] +
                         [cross(r, c) for r in ('012', '345', '678') for c in ('012', '345', '678')])
        units = dict((s, [u for u in list if s in u]) for s in self.variables)
        constraints = dict((s, set(sum(units[s], [])) - set([s])) for s in self.variables)
        return constraints




class CSP_Solver(object):
    """
    This class is used to solve the CSP with backtracking using the minimum value remaining heuristic.
    Implement functions in the backtracking sudo code in figure 6.5 in the text book.


    """
    def __init__(self, puzzle_file):
        '''
        Initialize the solver instance. The lower the number of the puzzle file the easier it is.
        It is a good idea to start with the easy puzzles and verify that a solution is correct manually.
        Run on the hard puzzles to make sure, it is not violating corner cases that come up.
        Harder puzzles will take longer to solve.
        :param puzzle_file: the puzzle file to solve
        '''
        self.sudoku = Sudoku(puzzle_file) # this line has to be here to initialize the puzzle
        self.num_guesses = 0
        self.count = 0
        vars = [(row, col) for row in range(9) for col in range(9)]
        domains = self.getDomain(vars)
        self.csp = CSP(vars, domains)

    def getDomain(self,vars):
        values = dict()
        for cell in vars:
            if self.sudoku.board[cell[0]][cell[1]] != 0:
                values[cell] = [self.sudoku.board[cell[0]][cell[1]]]
            else:
                values[cell] = [a for a in range(1,len(self.sudoku.board)+1)]
        return values

    ################################################################
    ### Test by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver = CSP_Solver('puz-001.txt')
    ### solved_board, num_guesses = csp_solver.solve()
    ### so `solve' method must return these two items.
    ################################################################
    def solve(self):
        '''
        This method solves the puzzle initialized in self.sudoku
        Define backtracking search methods that this function calls
        The return from this function NEEDS to match the correct type
        Return None, number of guesses no solution is found
        :return: tuple (list of list (ie [[]]), number of guesses
        '''


        print(self.sudoku.board_str())
        x, y = self.backtracking_search({}, self.csp)
        if (x==None):
            print("LOL")
            print(self.count)
        return x,y

    def backtracking_search(self, sudoku, csp):
        '''
        This function might be helpful to initialize a recursive backtracking search function

        :param sudoku: Sudoku class instance
        :param csp: CSP class instance
        :return: board state (list of lists), num guesses
        '''
        return self.recursive_backtracking(sudoku, csp), self.num_guesses

    def recursive_backtracking(self, sudoku, csp):
        '''
        recursive backtracking search function.
        :param sudoku: Sudoku class instance
        :param csp: CSP class instance
        :return: board state (list of lists)
        '''
        self.count= self.count+1
        if self.sudoku.complete():
            print(self.sudoku.board_str())
            print("==========================")
            return self.sudoku.board

        var = self.select_unassigned_var(sudoku)
        domain = deepcopy(csp.domains)

        for value in self.order_domain_values(var,self.csp):
            if self.consistent(var, value, sudoku, csp.constraints):
                sudoku[var] = value
                self.sudoku.board[var[0]][var[1]] = value

                self.num_guesses = self.num_guesses + 1
                if self.sudoku.complete():
                    return self.sudoku.board

                inferences = self.Inference(sudoku, csp, var, value)

                result = self.recursive_backtracking(sudoku, csp)
                if result != None:
                    return result

            del sudoku[var]
            self.sudoku.board[var[0]][var[1]] = 0
            csp.domains.update(domain)

        return None

    def Inference(self, sudoku, csp, var, value):
        for neighbor in csp.constraints[var]:
            if neighbor not in sudoku and value in csp.domains[neighbor]:
                csp.domains[neighbor].remove(value)




    # for neighbor in csp.constraints[var]:
    #     if board[neighbor[0]][neighbor[1]] == 0 and value in csp.domains[neighbor]:
    #
    #         if len(csp.domains[neighbor]) == 1:
    #             return "FAILURE"
    #
    #         csp.domains[neighbor].remove(value)
    #
    #         if len(csp.domains[neighbor]) == 1:
    #             flag = self.Inference(board, csp, neighbor, csp.domains[neighbor][0])
    #             if flag == "FAILURE":
    #                 return "FAILURE"
    #
    # return "SUCCESS"


    def select_unassigned_var(self, sudoku):
        '''
        Function that should select an unassigned variable to assign next
        :param board: list of lists
        :return: variable
        '''

        unassigned_variables = dict(
            (cell, len(self.csp.domains[cell])) for cell in self.csp.domains if cell not in sudoku.keys())
        mrv = min(unassigned_variables, key=unassigned_variables.get)
        return mrv
        # unassigned_variables = dict(
        #     (cell, len(self.csp.domains[cell])) for cell in self.csp.domains if board[cell[0]][cell[1]] == 0)
        # mrv = min(unassigned_variables, key=unassigned_variables.get)
        # return mrv #

    def order_domain_values(self, var, csp):
        '''
        A function to return domain values for a variable.
        :param var: variable
        :param csp: CSP problem instance
        :return: list of domain values for var
        '''
        return csp.domains[var]

    def consistent(self, var, value, sudoku, constraints):
        '''
        This function checks to see if assigning value to var on board violates any of the constraints
        :param var: variable to be assigned, tuple (row col)
        :param value: value to assign to var
        :param board: board state (list of list)
        :param constraints: to check to see if they are violated
        :return: True if consistent False otherwise
        '''
        for neighbor in constraints[var]:
            if neighbor in sudoku.keys() and sudoku[neighbor] == value:
                return False
        return True

        #
        # for neighbor in constraints[var]:
        #     if  board[neighbor[0]][neighbor[1]] == value:
        #         return False
        # return True

if __name__ == '__main__':
    csp_solver = CSP_Solver('puz-001.txt')
    
    for name in ['001', '002', '010', '015', '025', '026', '048', '051', '062', '076', '081', '082', '090', '095', '099', '100', 'test']:
        csp_solver = CSP_Solver('puz-'+ name + '.txt')
    
        solution, guesses = csp_solver.solve()
        print(csp_solver.sudoku.board_str())
        csp_solver.sudoku.write('puz-' + name + '-solved.txt')
        print(guesses)
    