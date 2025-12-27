
## Assignment Checklist

### Setup & Environment
- [ ] Run `bash setup.sh` successfully
- [ ] Verify model weights downloaded (stories42M.pt)

### Core Implementation
- [ ] **llama.py**
- [ ] **rope.py**
- [ ] **optimizer.py**
- [ ] **classifier.py**
- [ ] **lora.py**

### Testing & Validation
- [ ] Pass `python sanity_check.py`
- [ ] Pass `python optimizer_test.py` 
- [ ] Pass `python rope_test.py` 
- [ ] Generate coherent text with `python run_llama.py --option generate`
- [ ] Complete SST zero-shot prompting
- [ ] Complete CFIMDB zero-shot prompting  
- [ ] Complete SST fine-tuning
- [ ] Complete CFIMDB fine-tuning
- [ ] Complete SST LoRA fine-tuning
- [ ] Complete CFIMDB LoRA fine-tuning

### Advanced Features (Optional - A+)
- [ ] Other advanced techniques (see A+ options above)
- [ ] Write 1-2 page report documenting improvements

### Submission Preparation
- [ ] Generate all required output files
- [ ] Validate submission with `python3 prepare_submit.py`
- [ ] Verify file size < 10MB
- [ ] Create zip file with proper ANDREWID structure
- [ ] Include optional report and feedback files
- [ ] Final submission check before Canvas upload
  
