" Vim Editor Graphics
set number
set relativenumber
set tabstop=4
set shiftwidth=4
set autoindent
colorscheme slate

" Vimplug Automatic Download
let data_dir = has('nvim') ? stdpath('data') . '/site' : '~/.vim'
if empty(glob(data_dir . '/autoload/plug.vim'))
  silent execute '!curl -fLo '.data_dir.'/autoload/plug.vim --create-dirs  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
  autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif

"Vim Plugins
call plug#begin()
Plug 'lervag/vimtex' " VIMTeX plugin
call plug#end()

" VimTeX Configuration
filetype plugin indent on
syntax enable
let g:vimtex_view_general_viewer = 'sumatraPDF'
let g:vimtex_view_general_options = '-reuse-instance @pdf'
let g:vimtex_view_general_options_latexmk = '-reuse-instance'
