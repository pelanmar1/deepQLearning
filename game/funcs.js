

formatGrid = function (grid) {
    var newGrid = replaceValues(grid);
    newGrid = transposeGrid(newGrid);
    return newGrid;
}

replaceValues = function (grid) {
    var n = grid.length;
    var m = grid[0].length;

    for (var i = 0; i < n; i++) {
        for (var j = 0; j < m; j++) {
            if (grid[i][j] !== null) {
                grid[i][j] = grid[i][j].value;
            } else {
                grid[i][j] = 0;
            }
        }
    }
    return grid;

}
transposeGrid = function (grid) {
    var n = grid.length;
    var m = grid[0].length;
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < i; j++) {
            var a = grid[i][j];
            var b = grid[j][i];
            grid[i][j] = b;
            grid[j][i] = a;
        }
    }
    return grid;

}
isOver = function(){
    if (localStorage.hasOwnProperty("gameState")) {
        var over = JSON.parse(localStorage.gameState).over;
        return over;
    }
    return true;
}

getGrid = function () {
    if (localStorage.hasOwnProperty("gameState")) {
        var input = JSON.parse(localStorage.gameState).grid.cells;
        var grid = formatGrid(input);
        return grid;
    }
    return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
}
getScore = function () {
    var strScore = JSON.parse(localStorage.gameState).score;
    var score = strScore;
    return score;
}

getBestScore = function(){
    if(localStorage.hasOwnProperty("bestScore")){
        return localStorage.bestScore;
    }
    return "0";
}
