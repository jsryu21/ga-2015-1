FOR /L %%i IN (0, 1, 1) DO (
	Project1.exe 1 3 10 0 2 4 2 < cycle.in.318 >> out_318_%1.txt
)