FOR /L %%i IN (0, 1, 5) DO (
	FOR /L %%j IN (0, 1, 3) DO (
		echo %%i %%j >> result_3.txt
		Project1.exe 4 6 6 3 5 %%i %%j < cycle.in.318 >> result_3.txt
	)
)