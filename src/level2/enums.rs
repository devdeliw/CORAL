#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CoralTranspose{
    NoTranspose,
    Transpose,  
    ConjugateTranspose
}

#[derive(Debug, Copy, Clone)]
pub enum CoralTriangular { 
    UpperTriangular, 
    LowerTriangular 
}   

#[derive(Debug, Copy, Clone)]
pub enum CoralDiagonal {
    UnitDiagonal, 
    NonUnitDiagonal
}
