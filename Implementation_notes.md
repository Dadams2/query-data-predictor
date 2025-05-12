## Implementation Notes

I attempted using duckdb for reading and dealing with sdss data, unfortunately it struggles to read particularly large files and as such I will have to put out to a bigger database like postgres

In a previous implementation I used to normalise the interestingness rule score by the length of the measure. This has been forgeone in the current implementation in favor of larger interestingness scores for contribution

for the purposes of running experiments properly and doing decent visualisation intermeidate results are now written out to a file