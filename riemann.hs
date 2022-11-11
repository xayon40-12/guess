import System.Environment (getArgs)
-- ref: arXiv:nucl-th/9809044v1 14 Sep 1998, 
-- Dirk H. Rischke, Fluid Dynamics for Relativistic Nuclear Collisions

e0 = 10
emin = 1
cs2 = 1/3
cs = sqrt cs2
eps = 1e-10

v e = (1-(e/e0)**(2*cs/(1+cs2)))/(1+(e/e0)**(2*cs/(1+cs2)))
xt e = (v e - cs)/(1 - v e*cs)

cont (el, v') = sqrt $ c1*c1+c2*c2 
    where 
        vel = v el
        vl = (vel+vr)/(1+vel*vr)
        vr = -v'
        er = emin
        gl = 1/(1-vl**2)
        gr = 1/(1-vr**2)
        c1 = el*gl*vl-er*vr*gr
        c2 = 4*el*gl*vl*vl+el-(4*er*gr*vr*vr+er)
                
grad f (a, b) = (fab, da/d, db/d)
    where
        fab = f (a,b)
        da = f (a+eps,b)-fab
        db = f (a,b+eps)-fab
        d = sqrt (da*da+db*db)
        
next f (e, _, a, b) = (e*0.99, fab, a-e*da, b-e*db)
    where (fab, da,db) = grad f (a, b)

-- search the energy and speed of the chock front
(eq, vc) = search' 1e-1 ((e0+emin)/3) (0.5) 1
    where
        search' err e v i | i<eps = (e, v)
        search' err e v _ = let (nerr, fab, ne, nv) = next cont (err, 0, e, v) in search' nerr ne nv fab

f x | x < xt e0 = e0
    | x < xt eq = e0*((1-cs)/(1+cs)*(1-x)/(1+x))**((1+cs2)/(2*cs))
    | x < vc = eq
    | otherwise = emin

f2 x | x < xt e0 = e0
    | x < 1 = e0*((1-cs)/(1+cs)*(1-x)/(1+x))**((1+cs2)/(2*cs))
    | otherwise = emin
        
choice [] = f
choice ["void"] = f2

main :: IO ()
main = do
    args <- getArgs
    let fun = choice args
    getContents >>= mapM_ (print . fun . read) . lines
