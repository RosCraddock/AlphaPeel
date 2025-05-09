
# Changelog

## Contributors

* Audrey Martin
* augustusgrant (GitHub user)
* Gregor Gorjanc
* Ros Craddock
* Xinger Tang
* Yu Zhang


## AlphaPeel 1.1.8

* Fixed bug due to setuptools package being updated for all wheel file naming to follow binary distribution specification (i.e all lower case).

* Updated documentation for the bug.

## AlphaPeel 1.1.7

* allow metafounders, defined as “MF_”, in the pedigree file input.

* added ``alt_allele_prob_file`` to allow user-inputted alternative allele frequencies for each metafounder and loci. For now, these are restricted to be between 0.01 and 0.99.

* added main_metafounder to allow user to assign the default metafounder to use where a metafounder has not been assigned to a founder in the pedigree.

* added ``update_alt_allele_prob`` to allow the base alternative allele frequencies to be updated after each peeling cycle based on the mean of the founders within the assigned metafounder.

* User-warnings for metafounder implementation.

* Documentation updates for metafounder and estimation of alternative allele frequency.

* Functional and accuracy tests for metafounder implementation.

* Metafounder simulation code via AlphaSimR for testing.

* Updated reference to tinyhouse.

## AlphaPeel 1.1.6

* set default hap and geno threshold as 1/3 when calling genotypes.

* resolved bug to produce output file with ``-hap`` and ``-geno``.

* Addition of map file input for non-hybrid mode.

## AlphaPeel 1.1.5

* Updates in option and file names. These include:

  * ``no_dosages`` to ``no_dosage``
  * ``calling_threshold`` to ``geno_threshold``
  * ``call_phase`` to ``hap``
  * ``haps`` to ``phased_geno_prob``
  * ``pedigree`` to ``ped_file``
  * ``genotypes`` to ``geno_file``
  * and more, for all changes please visit: https://github.com/AlphaGenes/AlphaPeel/issues/113#issue-1935197000
 
* Addition of output options: ``seg_prob``, ``geno``, ``hap_threshold``, ``geno_prob``, ``est_rec_prob``.

* Updates the documentation and help functions.

* Updates to accuracy and functional tests for new argument names.

## AlphaPeel 1.1.4

* Implementation of functional and accuracy testing with pytest.

* Implementation of pre-committ protocol through Black and Flake8.

* Implementation of cross-platform tests workflow through github actions.

* Update theme for the HTML documentation.

* Added instructions for how to contribute to AlphaPeel.

* Fixed bug on the loading of submodule.

* Modified the URL for installation.
