f [x,y,t] = 2**(8/3)/(t**(4/3)*(1+2*(t**2+r**2)+(t**2-r**2)**2)**(4/3))
    where r = sqrt (x**2 + y**2)

main :: IO ()
main = getContents >>= mapM_ (print . f . map read . words) . lines
